-- internal library
local optim = require 'optim'
local path = require 'pl.path'

-- local library
local RNN = require 'model.my_RNN'
local model_utils = require 'util.model_utils'
local data_loader = require 'util.data_loader'
require 'util.misc'
local Top_Net = require 'model/Top_NN_Classifier'
local evaluate_process = require 'evaluate_process_v2'
local table_operation = require 'util/table_operation'
local define_my_model = require 'model/define_my_model_v2'

local train_process_v2 = {}

-- preprocessing helper function
function prepro(x)

    --if optgpuid >= 0 then
    x = x:float():cuda()
    --y = y:float():cuda()
    --end
    return x
end

--- process one batch to get the gradients for optimization and update the parameters
-- return the loss value of one minibatch of samples
local function feval(opt, loader, model, rmsprop_para, iter_count)
  -- decode the model and parameters,
  -- since it is just the reference to the same memory location, hence it is not time-consuming.

  local attention = model.attention
  local att_fusion = model.att_fusion
  local top_net = model.top_net
  local criterion = model.criterion
  local params_flat = model.params_flat
  local grad_params_flat = model.grad_params_flat
  local params_grad_all_batches = model.params_grad_all_batches


  ---------------------------- get minibatch --------------------------
  ---------------------------------------------------------------------

  local data_index = loader:get_next_train_batch(opt.batch_size)
  local loss_total = 0
  local x1,x2,x,x2_pad
  params_grad_all_batches:zero()

  -- Process the batch of samples one by one, since different sample contains different length of time series,
  -- hence it's not convenient to handle them together
  for batch = 1, opt.batch_size do
    local current_data_index = data_index[batch]
    if opt.multimodal then
      x1 = loader.train_X_visual[current_data_index]
      x2 = loader.train_X_motion[current_data_index]

      if opt.gpuid>= 0 then
        x1 = prepro(x1)
        x2 = prepro(x2)
      end

      if x1:dim() == 3 and x2:size(1) == 1 then
        x1 = x1:view(x1:size(2), x1:size(3))
        x2 = x2:view(x2:size(2), x2:size(3))
      end

    else
      x = loader.train_X_visual[current_data_index]
      if opt.gpuid>= 0 then
        x = prepro(x)
      end
      if x:dim() == 3 and x:size(1) == 1 then
        x = x:view(x:size(2), x:size(3))
      end
    end

    local true_y = loader.train_y[current_data_index]+1

    ---------------------- forward pass of the whole model -------------------
    --------------------------------------------------------------------------
    
    -- perform the forward pass for attention model_utils
    local attention_weights1, hidden_z_value1
    local attention_weights2, hidden_z_value2
    attention_weights1, hidden_z_value1 = attention.forward(x1, opt, 'training',1)
    attention_weights2, hidden_z_value2 = attention.forward(x2, opt, 'training',2)

    -- perform the forward for the top-net module
    local net_output = nil
    local hidden_rep1 = nil
    local hidden_rep2 = nil

    hidden_rep1 = top_net.forward(x1, attention_weights1, opt, 'training',1)
    hidden_rep2 = top_net.forward(x2, attention_weights2, opt, 'training',2)
    
    -- apply late fusion fusion
    local fused_scores = torch.zeros(2):cuda()
    local negative_mean = (hidden_rep1[1]*(-8) + hidden_rep2[1]*(-2))/10
    local positive_mean = (hidden_rep1[2]*(-8) + hidden_rep2[2]*(-2))/10

    fused_scores[1] = -negative_mean
    fused_scores[2] = -positive_mean

    -- net_output, x_fused = att_fusion.forward(hidden_rep1, hidden_rep2, opt)
    local current_loss = criterion:forward(fused_scores, true_y)
    local _, pred_label = fused_scores:squeeze():max(1)
    loss_total = loss_total + current_loss

    ---------------------- backward pass of the whole model ---------------------
    -----------------------------------------------------------------------------

    -- peform the backprop on the top_net
    grad_params_flat:zero()
    local grad_net1 = nil
    local grad_net2 = nil
    local grads1, grads2
    grads = criterion:backward(fused_scores, true_y)
    grad_net1 = top_net.backward(x1, attention_weights1, opt, grads, loader, 1)
    grad_net2 = top_net.backward(x2, attention_weights2, opt, grads, loader, 2)
    if opt.if_attention == 1 then
      attention.backward(opt, hidden_z_value1, grad_net1, x1, 1)
      attention.backward(opt, hidden_z_value2, grad_net2, x2, 2)
    end
    params_grad_all_batches:add(grad_params_flat)

  end
  loss_total = loss_total / opt.batch_size
  -- udpate all the parameters
  params_grad_all_batches:div(opt.batch_size)
  params_grad_all_batches:clamp(-opt.grad_clip, opt.grad_clip)
  if opt.opt_method == 'rmsprop' then
    local function feval_rmsprop(p)
      return loss_total, params_grad_all_batches
    end
    optim.rmsprop(feval_rmsprop, params_flat, rmsprop_para.config)
  elseif opt.opt_method  == 'adam' then
    local function feval_adam(p)
      return loss_total, params_grad_all_batches
    end
    local optim_config = {learningRate = opt.learning_rate}
    optim.adam(feval_adam, params_flat, optim_config)
  elseif opt.opt_method == 'gd' then -- 'gd' simple direct minibatch gradient descent
    params_flat:add(-opt.learning_rate, params_grad_all_batches)
  else
    error("there is no such optimization option!")
  end

  return loss_total
end

function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

--- major functionf
function train_process_v2.train(opt)


  local testPath = 'data/test_mfcc/'
  local test_clips_ = ls(testPath)

  test_clips = {}

  for j=1, #test_clips_ do
    local videoPath = testPath..test_clips_[j]
    if string.find(videoPath,"violence") then
      table.insert(test_clips,videoPath)
    end
  end

  ------------------- create the data loader class ----------
  -----------------------------------------------------------

  local loader = data_loader.create(opt)
  local do_random_init = true

  -- local iterations = math.floor(opt.max_epochs * loader.nTrain / opt.batch_size)
  local iterations = opt.num_iterations
  local train_losses = torch.zeros(iterations)
  local timer = torch.Timer()
  local time_s = timer:time().real
  local epoch = 0
  local better_times_total = 0
  local better_times_decay = 0
  local current_best_acc = 0
  local whole_validation_loss,  validation_acc = nil

  if opt.cross_val==1 then

    local nSplits = loader.nSplits
    local accuracies = {}

    for j=1,nSplits do

      print("Building model ...")
      ------------------ begin to define the whole model --------------------------
      -----------------------------------------------------------------------------
      local model = {}
      model, opt = define_my_model.define_model(opt, loader)

        --------------- start optimization here -------------------------
      -----------------------------------------------------------------
      -- for rmsprop
      local rmsprop_config = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
      local rmsprop_state = {}
      local rmsprop_para = {config = rmsprop_config, state = rmsprop_state}

      -- cross validation
      print("Split : ", j)
      -- loading the split data_loader
      data = loader.data[j]

      loader.train_X_visual = data.train_X_visual
      loader.train_X_motion = data.train_X_motion
      loader.test_X_visual = data.test_X_visual
      loader.test_X_motion = data.test_X_motion

      loader.train_y = data.train_y_visual
      loader.valid_y = data.test_y_visual

      loader.nTrain = loader.train_y:size(1)
      loader.rand_order = torch.randperm(loader.nTrain)
      loader.nTest = loader.valid_y:size(1)
      print("Data train len : ", loader.train_X_visual:size(1))

      local epoch = 0

      for i = 1, iterations do
        epoch = i / loader.nTrain * opt.batch_size
        local time_ss = timer:time().real
        -- optimize one batch of training samples
        train_losses[i] = feval(opt, loader, model, rmsprop_para, i)
        -- local whole_train_loss, train_acc = evaluate_process.evaluate_set('test', opt, loader, model)
        -- print("accuracy: ", train_acc)
        local time_ee = timer:time().real
        local time_current_iteration = time_ee - time_ss
        if i % 10 == 0 then collectgarbage() end

        if i % opt.print_every == 0 then
          print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, time/batch = %.4fs",
            i, iterations, epoch, train_losses[i], time_current_iteration))
        end

        if i % opt.evaluate_every == 0 then
          local temp_sum_loss = torch.sum(train_losses:sub(i - opt.evaluate_every/opt.batch_size+1, i))
          local temp_mean_loss = temp_sum_loss / opt.evaluate_every * opt.batch_size
          print(string.format('average loss in the last %d iterations = %6.8f', opt.evaluate_every, temp_mean_loss))
          print('learning rate: ', opt.learning_rate)
    
          local whole_validation_loss,  validation_acc = nil
          if opt.validation_size == 0 then
            local whole_train_loss, train_acc = evaluate_process.evaluate_set('train', opt, loader, model)
            whole_validation_loss = whole_train_loss
            validation_acc = train_acc
          else
            whole_validation_loss,  validation_acc = evaluate_process.evaluate_set('test', opt, loader, model)
          end
          local whole_test_loss, test_acc 
          if opt.if_output_step_test_error then
            whole_test_loss, test_acc = evaluate_process.evaluate_set('test', opt, loader, model)
          end
          local time_e = timer:time().real
          print(string.format('elasped time in the last %d iterations: %.4fs,    total elasped time: %.4fs', 
            opt.evaluate_every, time_e-time_s, time_e))
          if validation_acc > current_best_acc then
            current_best_acc = validation_acc
            better_times_total = 0
            better_times_decay = 0
            --- save the current trained best model
            define_my_model.save_model(opt, model)
            if validation_acc == 0 then
              break
            end
          else
            better_times_total = better_times_total + 1
            better_times_decay = better_times_decay + 1
            if better_times_total >= opt.stop_iteration_threshold then
              print(string.format('no more better result in %d iterations! hence stop the optimization!', 
                opt.stop_iteration_threshold))
              break
            elseif better_times_decay >= opt.decay_threshold then
              print(string.format('no more better result in %d iterations! hence decay the learning rate', 
                opt.decay_threshold))
              local decay_factor = opt.learning_rate_decay
              rmsprop_config.learningRate = rmsprop_config.learningRate * decay_factor -- decay it
              opt.learning_rate = rmsprop_config.learningRate -- update the learning rate in opt
              print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. rmsprop_config.learningRate)
              better_times_decay = 0 
              -- back to the currently optimized point
              -- print('back to the currently best optimized point...')
              -- model = define_my_model.load_model(opt, model, false)
            end
          end     
          print('better times: ', better_times_total, '\n\n')
        end
      end
      -- test evaluation
      local whole_test_loss, test_acc

      -- whole_test_loss, test_acc = evaluate_process.evaluate_set('test', opt, loader, model, j, -1)
      -- accuracies[j]= test_acc
      accuracies[j] = current_best_acc
      current_best_acc=0
    end

    -- obtain the avg acc
    local sum = 0
    local avg = 0
    for i=1,nSplits do
      print(accuracies[i])
      sum = sum + accuracies[i]
    end

    avg = sum / nSplits
    print("Average Accuracy : ", avg)

  else
    print("Building model ...")
    ------------------ begin to define the whole model --------------------------
    -----------------------------------------------------------------------------
    local model = {}
    model, opt = define_my_model.define_model(opt, loader)

      --------------- start optimization here -------------------------
    -----------------------------------------------------------------
    -- for rmsprop
    local rmsprop_config = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
    local rmsprop_state = {}
    local rmsprop_para = {config = rmsprop_config, state = rmsprop_state}

    data = loader.data
    loader.train_X_visual = data.train_X_visual
    loader.train_X_motion = data.train_X_motion
    loader.test_X_visual = data.test_X_visual
    loader.test_X_motion = data.test_X_motion
    loader.train_y = data.train_y_visual
    loader.valid_y = data.test_y_visual

    loader.nTrain = loader.train_y:size(1)
    loader.rand_order = torch.randperm(loader.nTrain)
    loader.nTest = loader.valid_y:size(1)
    print("Data train len : ", loader.train_X_visual:size(1))

    local epoch = 0

    for i = 1, iterations do
      epoch = i / loader.nTrain * opt.batch_size
      local time_ss = timer:time().real
      -- optimize one batch of training samples
      train_losses[i] = feval(opt, loader, model, rmsprop_para, i)
      -- local whole_train_loss, train_acc = evaluate_process.evaluate_set('test', opt, loader, model)
      -- print("accuracy: ", train_acc)
      local time_ee = timer:time().real
      local time_current_iteration = time_ee - time_ss
      if i % 10 == 0 then collectgarbage() end

      if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, time/batch = %.4fs",
          i, iterations, epoch, train_losses[i], time_current_iteration))
      end

      if i % opt.evaluate_every == 0 then
        local temp_sum_loss = torch.sum(train_losses:sub(i - opt.evaluate_every/opt.batch_size+1, i))
        local temp_mean_loss = temp_sum_loss / opt.evaluate_every * opt.batch_size
        print(string.format('average loss in the last %d iterations = %6.8f', opt.evaluate_every, temp_mean_loss))
        print('learning rate: ', opt.learning_rate)
  
        local whole_validation_loss,  validation_acc = nil
        if opt.validation_size == 0 then
          local whole_train_loss, train_acc = evaluate_process.evaluate_set('train', opt, loader, model)
          whole_validation_loss = whole_train_loss
          validation_acc = train_acc
        else
          whole_validation_loss,  validation_acc = evaluate_process.evaluate_set('test', opt, loader, model)
        end
        local whole_test_loss, test_acc 
        if opt.if_output_step_test_error then
          whole_test_loss, test_acc = evaluate_process.evaluate_set('test', opt, loader, model)
        end
        local time_e = timer:time().real
        print(string.format('elasped time in the last %d iterations: %.4fs,    total elasped time: %.4fs', 
          opt.evaluate_every, time_e-time_s, time_e))
        if validation_acc > current_best_acc then
          current_best_acc = validation_acc
          better_times_total = 0
          better_times_decay = 0
          --- save the current trained best model
          define_my_model.save_model(opt, model)
          if validation_acc == 0 then
            break
          end
        else
          better_times_total = better_times_total + 1
          better_times_decay = better_times_decay + 1
          if better_times_total >= opt.stop_iteration_threshold then
            print(string.format('no more better result in %d iterations! hence stop the optimization!', 
              opt.stop_iteration_threshold))
            break
          elseif better_times_decay >= opt.decay_threshold then
            print(string.format('no more better result in %d iterations! hence decay the learning rate', 
              opt.decay_threshold))
            local decay_factor = opt.learning_rate_decay
            rmsprop_config.learningRate = rmsprop_config.learningRate * decay_factor -- decay it
            opt.learning_rate = rmsprop_config.learningRate -- update the learning rate in opt
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. rmsprop_config.learningRate)
            better_times_decay = 0 
            -- back to the currently optimized point
            -- print('back to the currently best optimized point...')
            -- model = define_my_model.load_model(opt, model, false)
          end
        end     
        print('better times: ', better_times_total, '\n\n')
      end

    end
    -- test evaluation
    local whole_test_loss, test_acc

    whole_test_loss, test_acc = evaluate_process.evaluate_set('test', opt, loader, model, -1)
    print("Test Accuracy : ", test_acc)
  end

  local time_e = timer:time().real
  print('total elapsed time:', time_e)
end

return train_process_v2

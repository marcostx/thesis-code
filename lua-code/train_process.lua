-- internal library
local optim = require 'optim'
local path = require 'pl.path'

-- local library
local RNN = require 'model.my_RNN'
local model_utils = require 'util.model_utils'
local data_loader = require 'util.data_loader'
require 'util.misc'
local Top_Net = require 'model/Top_NN_Classifier'
local evaluate_process = require 'evaluate_process'
local table_operation = require 'util/table_operation'
local define_my_model = require 'model/define_my_model'

local train_process = {}

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
  local x1,x2,x
  params_grad_all_batches:zero()

  -- Process the batch of samples one by one, since different sample contains different length of time series,
  -- hence it's not convenient to handle them together
  for batch = 1, opt.batch_size do
    local current_data_index = data_index[batch]
    if opt.multimodal then
      x1 = loader.train_X_visual[current_data_index]
      x2 = loader.train_X_motion[current_data_index]
      
      x1 = prepro(x1)
      x2 = prepro(x2)
      
      if x1:dim() == 3 and x1:size(1) == 1 then
        x1 = x1:view(x1:size(2), x1:size(3))
        x2 = x2:view(x2:size(2), x2:size(3))
      end
    else
      x = loader.train_X[current_data_index]
      x = prepro(x)
      if x:dim() == 3 and x:size(1) == 1 then
        --print(x:size(1),x:size(2), x:size(3)) 
        x = x:view(x:size(2), x:size(3))
      end
    end

    local true_y = loader.train_y[current_data_index]+1
    -- elseif x:dim() > 3 then
    --   error('x:dim > 3')
    -- end
    local x_length
    if opt.multimodal then
      if opt.early then
        x_length = x1:size(2) + x2:size(2)
      else
        x_length = x1:size(2)
      end
    else
      x_length = x:size(2)
    end

    --x, true_y = prepro(x, true_y)

    ---------------------- forward pass of the whole model -------------------
    --------------------------------------------------------------------------
    if opt.multimodal then
      if opt.early then
        x = torch.cat(x1, x2, 1)
      else
        x = att_fusion.forward(x1, x2, opt)
      end
    end

    -- perform the forward pass for attention model_utils
    local attention_weights, hidden_z_value
    if opt.if_attention == 1 then
      attention_weights, hidden_z_value = attention.forward(x, opt, 'training')
      -- print(attention_weights)
    else
      attention_weights = torch.ones(x_length)
    end

    -- perform the forward for the top-net module
    local net_output = nil
    if opt.top_c == 'NN' then
      if opt.if_original_feature == 1 then
        net_output = top_net:forward({x, attention_weights})
      else
        net_output = top_net:forward({hidden_z_value, attention_weights})
      end
    elseif opt.top_c == 'lstm' or opt.top_c == 'rnn' or opt.top_c == 'gru' or opt.top_c == 'TAGM' then
      net_output = top_net.forward(x, attention_weights, opt, 'training')
    else
      error('no such top classifier!')
    end
    --print(net_output)
    --print(true_y)
    --compute the loss
    --  local current_loss = criterion:forward(net_output, torch.Tensor({true_y})) -- for batch_size == 1
    local current_loss = criterion:forward(net_output, true_y)
    local _, pred_label = net_output:squeeze():max(1)
    loss_total = loss_total + current_loss

    --print(pred_label, true_y)

    ---------------------- backward pass of the whole model ---------------------
    -----------------------------------------------------------------------------

    -- peform the backprop on the top_net
    grad_params_flat:zero()
    local grad_net = nil
    if opt.top_c == 'NN' then
      if opt.if_original_feature ==1 then
        grad_net = top_net:backward({x, attention_weights}, criterion:backward(net_output, true_y))
        if opt.if_attention == 1 then
          attention.backward(opt, hidden_z_value, grad_net[2], x)
        end
      else
        grad_net = top_net:backward({hidden_z_value, attention_weights}, criterion:backward(net_output, true_y))
        if opt.if_attention == 1 then
          attention.backward(opt, hidden_z_value, grad_net[2], x, grad_net[1])
        end
      end
    elseif opt.top_c == 'lstm' or opt.top_c == 'rnn' or opt.top_c == 'gru' then
      grad_net = top_net.backward(x, attention_weights, opt, criterion:backward(net_output, true_y), loader)
      if opt.if_attention == 1 then
        attention.backward(opt, hidden_z_value, grad_net[2], x)
      end
    elseif opt.top_c == 'TAGM' then
      grad_net = top_net.backward(x, attention_weights, opt, criterion:backward(net_output, true_y), loader)
      if opt.if_attention == 1 then
        attention.backward(opt, hidden_z_value, grad_net, x)
      end
      if opt.multimodal and opt.early == false then
        att_fusion.backward(opt, x1, x2, grad_net, loader)
      end
    else
      error('no such classifier!')
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
  elseif opt.opt_method == 'gd' then -- 'gd' simple direct minibatch gradient descent
    params_flat:add(-opt.learning_rate, params_grad_all_batches)
  elseif opt.opt_method  == 'adam' then
    local function feval_adam(p)
      return loss_total, params_grad_all_batches
    end
    local optim_config = {learningRate = opt.learning_rate}
    optim.adam(feval_adam, params_flat, optim_config)
  else
    error("there is no such optimization option!")
  end

  return loss_total
end

--- major function
function train_process.train(opt)

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
      loader.train_X = data.train_X
      loader.valid_X = data.valid_X
      loader.train_y = data.train_y
      loader.valid_y = data.valid_y

      loader.nTrain = loader.train_y:size(1)
      loader.rand_order = torch.randperm(loader.nTrain)
      loader.nTest = loader.valid_y:size(1)
      print("Data train len : ", loader.train_X:size(1))

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
          
          whole_validation_loss,  validation_acc = evaluate_process.evaluate_set('test', opt, loader, model)
          
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

      accuracies[j]= current_best_acc
      current_best_acc=0
    end

    -- obtain the avg acc
    local sum = 0
    local avg = 0
    for i=1,nSplits do
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

    loader.train_X = loader.data.train_X[{{}, {},{},{1,149}}]
    loader.valid_X = loader.data.valid_X[{{}, {},{},{1,149}}]
    loader.train_y = loader.data.train_y
    loader.valid_y = loader.data.valid_y
    
    loader.nTrain = loader.train_y:size(1)
    loader.rand_order = torch.randperm(loader.nTrain)
    loader.nTest = loader.valid_y:size(1)
    print("Data train len : ", loader.train_X:size(1))

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
      --print(i, opt.evaluate_every)
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

return train_process

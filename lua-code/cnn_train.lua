-- internal library
local optim = require 'optim'
local path = require 'pl.path'

-- local library
local RNN = require 'model.my_RNN'
local model_utils = require 'util.model_utils'
local data_loader = require 'util.data_loader'
require 'util.misc'
local evaluate_process = require 'cnn_evaluate'
local table_operation = require 'util/table_operation'
local define_my_model = require 'model/define_my_model_v2'

local train_process = {}

-- preprocessing helper function
function prepro(x)

    --if optgpuid >= 0 then
    x = x:cuda()
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
  local convlstm = model.convlstm
  local criterion = model.criterion
  local params_flat = model.params_flat
  local grad_params_flat = model.grad_params_flat
  local params_grad_all_batches = model.params_grad_all_batches
  local cnn_encoder = model.cnn_encoder
  local rnn_decoder = model.rnn_decoder


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
      x1 = loader.trainX[current_data_index]
      x2 = loader.train_X_motion[current_data_index]

      if opt.gpuid>= 0 then
        x1 = prepro(x1)
        x2 = prepro(x2)
      end

      if x1:dim() == 3 and x1:size(1) == 1 then
        x1 = x1:view(x1:size(2), x1:size(3))
        x2 = x2:view(x2:size(2), x2:size(3))
      end

    else
      x = loader.trainX[current_data_index]
      if opt.gpuid>= 0 then
        x = prepro(x)
      end
      if x:dim() == 3 and x:size(1) == 1 then
        x = x:view(x:size(2), x:size(3))
      end
    end


    local true_y = loader.trainY[current_data_index]+1
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
      x_length = x:size(4)
    end

    local cnn_embeddings=nil
    local attention_weights, hidden_z_value
    local l1_
    local l2_
    local l3_
    local conv_out_
    local fc1_
    local fc1_l1_
    local fc1_l2_
    local fc1_l3_
    local g

    --x, true_y = prepro(x, true_y)

    ---------------------- forward pass of the whole model -------------------
    --------------------------------------------------------------------------
    -- if opt.multimodal then
    --   if opt.early then
    --     x = torch.cat(x1, x2, 1)
    --   else
    --     x = att_fusion.forward(x1, x2, opt)
    --   end
    -- end

    -- -- perform the forward pass for attention model_utils

    -- perform the forward for the top-net module
    local net_output = nil
    if opt.top_c == 'NN' then
      if opt.if_original_feature == 1 then
        net_output = top_net:forward({x, attention_weights})
      else
        net_output = top_net:forward({hidden_z_value, attention_weights})
      end
    elseif opt.top_c == 'lstm' or opt.top_c == 'rnn' or opt.top_c == 'gru' or opt.top_c == 'TAGM' or opt.top_c == 'tsam' then
      net_output = top_net.forward(x, attention_weights, opt, 'training')
    elseif opt.top_c == 'cnn' then
      net_output = top_net.forward(x, opt, 'training')
    elseif opt.top_c == 'vgg_att' then
      -- net_output,l1_, l2_, l3_, conv_out_, fc1_, fc1_l1_,fc1_l2_,fc1_l3_, g = top_net.forward(x, opt, 'training')
      -- net_output = cnn.forward(x, opt, 'training')
      cnn_embeddings,l1_, l2_, l3_, conv_out_, fc1_, g = cnn_encoder.forward(x, opt, 'training')
      attention_weights, hidden_z_value = attention.forward(cnn_embeddings, opt, 'training')
      net_output = rnn_decoder.forward(cnn_embeddings,attention_weights, opt, 'training')
    elseif opt.top_c == 'convlstm' then
      cnn_embeddings = cnn_encoder.forward(x, opt, 'training')
      if opt.if_attention == 1 then
        attention_weights, hidden_z_value = attention.forward(cnn_embeddings, opt, 'training')
      else
        attention_weights = torch.ones(x_length)
      end
      net_output = rnn_decoder.forward(cnn_embeddings,attention_weights, opt, 'training')
    elseif opt.top_c == 'convlstm_v2' then
      net_output = convlstm.forward(x, opt, 'training')
    else
      error('no such top classifier!')
    end
    --compute the loss

    --  local current_loss = criterion:forward(net_output, torch.Tensor({true_y})) -- for batch_size == 1
    local current_loss = criterion:forward(net_output, true_y)
    local _, pred_label = net_output:squeeze():max(1)
    loss_total = loss_total + current_loss

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
    elseif opt.top_c == 'lstm' or opt.top_c == 'rnn' or opt.top_c == 'gru' or opt.top_c == 'tsam' then
      grad_net = top_net.backward(x, attention_weights, opt, criterion:backward(net_output, true_y), loader)
      if opt.if_attention == 1 then
        attention.backward(opt, hidden_z_value, grad_net, x)
      end
    elseif opt.top_c == 'TAGM' then
      grad_net = top_net.backward(x, attention_weights, opt, criterion:backward(net_output, true_y), loader)
      if opt.if_attention == 1 then
        attention.backward(opt, hidden_z_value, grad_net, x)
      end
      if opt.multimodal and opt.early == false then
        att_fusion.backward(opt, x1, x2, grad_net, loader)
      end
    elseif opt.top_c == 'cnn' then
      grad_net = top_net.backward(x, opt, criterion:backward(net_output, true_y), loader)
    elseif opt.top_c == 'vgg_att' then
      -- grad_net = top_net.backward(x, opt, criterion:backward(net_output, true_y))
      grad_net = rnn_decoder.backward(cnn_embeddings,attention_weights, opt, criterion:backward(net_output, true_y), loader)
      attention.backward(opt, hidden_z_value, grad_net, cnn_embeddings)
      cnn_encoder.backward(x, l1_, l2_, l3_, conv_out_, fc1_, g, opt, grad_net)
    elseif opt.top_c == 'convlstm' then
      grad_net = rnn_decoder.backward(cnn_embeddings,attention_weights, opt, criterion:backward(net_output, true_y),loader)
      if opt.if_attention == 1 then
        attention.backward(opt, hidden_z_value, grad_net, cnn_embeddings)
      end
      -- cnn_encoder.backward(x, opt, grad_net)
    elseif opt.top_c == 'convlstm_v2' then
      convlstm.backward(x, opt, criterion:backward(net_output, true_y))
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

  loader.nTrain = loader.trainY:size(1)
  loader.rand_order = torch.randperm(loader.nTrain)
  -- loader.nTest = loader.testY:size(1)
  print("Data train len : ", loader.trainX:size(1))

  local epoch = 0

  for i = 1, iterations do
    epoch = i / loader.nTrain * opt.batch_size
    local time_ss = timer:time().real
    -- optimize one batch of training samples
    train_losses[i] = feval(opt, loader, model, rmsprop_para, i)
    -- print("accuracy: ", train_acc)
    local time_ee = timer:time().real
    local time_current_iteration = time_ee - time_ss
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    local function isnan(x) return x ~= x end
    if isnan(train_losses[i]) then
      print('loss is NaN.  This usually indicates a bug.' ..
        'Please check the issues page for existing issues, or create a new issue, ' ..
        'if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
      break -- halt
    end

    -- check if the loss value blows up
    local function is_blowup(loss_v)
      if loss_v > opt.blowup_threshold then
        print('loss is exploding, aborting:', loss_v)
        return true
      else
        return false
      end
    end
    if is_blowup(train_losses[i]) then
      break
    end

    -- print training
    if i % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, time/batch = %.4fs",
        i, iterations, epoch, train_losses[i], time_current_iteration))
    end

    -- evaluation step
    if i % opt.evaluate_every == 0 then
      local temp_sum_loss = torch.sum(train_losses:sub(i - opt.evaluate_every/opt.batch_size+1, i))
      local temp_mean_loss = temp_sum_loss / opt.evaluate_every * opt.batch_size
      print(string.format('average loss in the last %d iterations = %6.8f', opt.evaluate_every, temp_mean_loss))
      print('learning rate: ', opt.learning_rate)

      local whole_test_loss, test_acc
      if opt.if_output_step_test_error then
        whole_test_loss, test_acc = evaluate_process.evaluate_set('test', opt, loader, model)
      end

      if test_acc > current_best_acc then
        current_best_acc = test_acc
        better_times_total = 0
        better_times_decay = 0
        --- save the current trained best model
        define_my_model.save_model(opt, model)
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
          print('back to the currently best optimized point...')
          model = define_my_model.load_model(opt, model, false)
        end
      end
      print('better times: ', better_times_total, '\n\n')
      -- save to log file
      local temp_file = nil
      temp_file = io.open(string.format('results_.txt',
        opt.current_result_dir), "a")

      temp_file:write('better times: ', better_times_total, '\n')
      temp_file:write('learning rate: ', opt.learning_rate, '\n')
      temp_file:write(string.format("%d/%d (epoch %.3f) \n", i, iterations, epoch))
      temp_file:write(string.format('average loss in the last %d (%5d -- %5d) iterations = %6.8f \n',
        opt.evaluate_every/opt.batch_size, i-opt.evaluate_every/opt.batch_size+1, i, temp_mean_loss))
      --      temp_file:write(string.format('train set loss = %6.8f, train age mean absolute error= %6.8f\n',
      --       whole_train_loss, differ_avg_train ))
      temp_file:write(string.format('test set loss = %6.8f, test accuracy = %6.8f\n',
        whole_test_loss, test_acc ))

      temp_file:write(string.format('\n'))
      temp_file:close()
    end
  end
  -- test evaluation
  local whole_test_loss, test_acc

  whole_test_loss, test_acc = evaluate_process.evaluate_set('test', opt, loader, model, -1, test_clips)
  print("Test Accuracy : ", test_acc)


  local time_e = timer:time().real
  print('total elapsed time:', time_e)
end

return train_process

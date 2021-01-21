
require 'image'

local path = require 'pl.path'
local AUC_EER = require 'util/my_AUC_EER_calculation'
require 'util.misc'
local data_loader = require 'util.data_loader'
local model_utils = require 'util.model_utils'
local define_my_model = require 'model.define_my_model'
local table_operation = require 'util/table_operation'

function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

local evaluate_process = {}

--- preprocessing helper function
local function prepro(x)
  x = x:float():cuda()
  return x
end

local function qualitative_analysis( model, opt, x, set_class, split, vid_id, attention_weights)

  temp_file = io.open(string.format("data/movies_seqlen30/%d/split_%d_testvideo_%d.csv", split,split, vid_id), "w")
  temp_file:write('attention scores, index \n')
  -- image.display(original)

  local attention_w = {}
  for i=1,attention_weights:size(1) do
    tmp = attention_weights[i]
    temp_file:write(string.format("%f, %d \n", tmp, i))
  end
  temp_file:close()

end


--- inference one sample¡
local function inference(model, x1, x2, true_y, opt)

  -- decode the model and parameters
  local attention = model.attention
  local top_net = model.top_net
  local att_fusion = model.att_fusion
  local criterion = model.criterion
  local params_flat = model.params_flat

  -- perform the forward pass for attention model
  local attention_weights1, hidden_z_value1
  local attention_weights2, hidden_z_value2
  attention_weights1, hidden_z_value1 = attention.forward(x1, opt, 'test',1)
  attention_weights2, hidden_z_value2 = attention.forward(x2, opt, 'test',2)

  -- perform the forward for the top-net module
  local hidden_rep1 = nil
  local hidden_rep2 = nil

  hidden_rep1 = top_net.forward(x1, attention_weights1, opt, 'test',1)
  hidden_rep2 = top_net.forward(x2, attention_weights2, opt, 'test',2)
  
  -- apply late fusion fusion
  local net_output = torch.zeros(2):cuda()
  local negative_mean = (hidden_rep1[1]*(-8) + hidden_rep2[1]*(-2))/10
  local positive_mean = (hidden_rep1[2]*(-8) + hidden_rep2[2]*(-2))/10

  net_output[1] = -negative_mean
  net_output[2] = -positive_mean

  --compute the loss
  --  local current_loss = criterion:forward(net_output, torch.Tensor({true_y})) -- for batch_size == 1
  local current_loss = criterion:forward(net_output, true_y)
  local _, pred_label = net_output:squeeze():max(1)

  return current_loss, pred_label:squeeze(), attention_weights
end

-- get the prediction and confidence score for one sample
local function predict(model, x1, x2,  opt)

  -- decode the model and parameters
  local attention = model.attention
  local top_net = model.top_net
  local criterion = model.criterion
  local params_flat = model.params_flat

  -- perform the forward pass for attention model
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
  -- kp km`
  --compute the loss
  local _, pred_label = fused_scores:squeeze():max(1)
  local confidence = fused_scores[2]

  return confidence,  pred_label:squeeze(), attention_weights1
end


--- for the gradient check
function evaluate_process.grad_check(model, x1, x2, true_y, opt)
  -- decode the model and parameters
  if opt.if_attention == 0 then
    model.attention.params_size = 1
  end
  
  local attention_params_flat = model.params_flat:sub(1, model.attention.params_size)
--  local attention_top_params_flat = model.params_flat:sub(model.attention.params_size-opt.rnn_size*2, model.attention.params_size)
--  local attention_grad_top_params_flat = model.grad_params_flat:sub(model.attention.params_size-opt.rnn_size*2, model.attention.params_size)
  local attention_grad_params_flat = model.grad_params_flat:sub(1, model.attention.params_size)
  print(model.params_flat:size())
  local top_net_params_flat = model.params_flat:sub(model.attention.params_size+1, -1)
  local top_net_grad_flat = model.grad_params_flat:sub(model.attention.params_size+1, -1)
  local total_params = model.params_size
  local function calculate_loss()
    local current_loss = inference(model, x, true_y, opt)
    return current_loss
  end  

  local function gradient_compare(params, grad_params)
    local check_number = math.min(200, params:nElement())
    local loss_minus_delta, loss_add_delta, grad_def
    if opt.gpuid >= 0 then
      loss_minus_delta = torch.CudaTensor(check_number)
      loss_add_delta = torch.CudaTensor(check_number)
      grad_def = torch.CudaTensor(check_number)
    else
      loss_minus_delta = torch.DoubleTensor(check_number)
      loss_add_delta = torch.DoubleTensor(check_number)
      grad_def = torch.DoubleTensor(check_number)    
    end
    local params_backup = params:clone()
    local rand_ind = torch.randperm(params:nElement())
    rand_ind = rand_ind:sub(1, check_number)
    for k = 3, 8 do
      local delta = 1 / torch.pow(1e1, k)
      print('delta:', delta)
      for i = 1, check_number do
        local ind = rand_ind[i]
        params[ind] = params[ind] - delta
        loss_minus_delta[i] = calculate_loss() 
        params[ind] = params[ind] + 2*delta
        loss_add_delta[i] = calculate_loss()
        local gradt = (loss_add_delta[i] - loss_minus_delta[i]) / (2*delta)
        grad_def[i] = gradt
        params[ind] = params[ind] - delta -- retore the parameters
        if i % 100 ==0 then
          print(i, 'processed!')
        end
      end
      params:copy(params_backup) -- retore the parameters
      local grad_model = grad_params:index(1, rand_ind:long())
      local if_print = true
      local threshold = 1e-4
      local inaccuracy_num = 0
      local reversed_direction = 0
      assert(grad_def:nElement()==grad_model:nElement())
      local relative_diff = torch.zeros(grad_def:nElement())
      relative_diff = torch.abs(grad_def - grad_model)
      relative_diff:cdiv(torch.cmax(torch.abs(grad_def), torch.abs(grad_model)))
      for i = 1, grad_def:nElement() do
        if if_print then
          print(string.format('index: %4d, rand_index: %4d, relative_diff: %6.5f,  gradient_def: %6.25f,  grad_model: %6.25f',
            i, rand_ind[i], relative_diff[i], grad_def[i], grad_model[i]))
        end
        if relative_diff[i] > threshold then
          if math.max(math.abs(grad_def[i]), math.abs(grad_model[i])) > 1e-8 then
            inaccuracy_num = inaccuracy_num + 1
          end   
        end
      end
      for i = 1, grad_def:nElement() do
        if grad_def[i] * grad_model[i] < 0 then
          if if_print then
            print(string.format('index: %4d, relative_diff: %6.5f,  gradient_def: %6.10f,  grad_params: %6.10f',
              i, relative_diff[i], grad_def[i], grad_model[i]))
          end
          reversed_direction = reversed_direction + 1
        end
      end

      print('there are', inaccuracy_num, 'inaccuracy gradients.')
      print('there are', reversed_direction, 'reversed directions.')
    end
  end


--     check rnn params
  gradient_compare(attention_params_flat, attention_grad_params_flat)  
--          gradient_compare(attention_top_params_flat, attention_grad_top_params_flat)  
--  --  --   check top_net params
--          gradient_compare(top_net_params_flat, top_net_grad_flat)


end

--- input @data_set is a data sequence (table of Tensor) to be evaluated
local function evaluation_set_mediaeval(opt, model, data_sequence_v1, data_sequence_v2)
  local l
  local accuracy = 0
  local total_pos = 0
  local data_size = data_sequence_v1:size(1)
  local all_attention_weights = {}
  local predictions = torch.zeros(data_size)
  -- local testPath = 'datasets/test/*.mp4'
  -- local trainPath = 'datasets/train/*.mp4'
  -- local test_cli=ps = ls(testPath)
  -- local train_clips = ls(trainPath)

  -- temp_file = io.open("mediaeval_test_predictions.txt", "w")
  temp_file = io.open("mediaeval_2015_test_predictions_late_fusion.txt", "w")
  local test_names_file = io.open("test_mediaeval_raw.txt", "r")
  local test_clips = {}
  for line in test_names_file:lines() do
    table.insert (test_clips, line);
  end

  for i = 1, data_size do
    local x, true_y
    x1 = data_sequence_v1[i]
    x2 = data_sequence_v2[i]
    -- clip_name = string.gsub(test_clips[i], "(.*/)(.*)", "%2")
    -- clip_name = string.sub(clip_name, 1, -5)
    clip_name = test_clips[i]
    if x1:dim() == 3 and x2:size(1) == 1 then
      x1 = x1:view(x1:size(2), x1:size(3))
      x2 = x2:view(x2:size(2), x2:size(3))
    end
    x1 = prepro(x1)
    x2 = prepro(x2)

    local confidence, predict_label, attention_weights = predict(model, x1, x2, opt)

    if predict_label == 2 then
      l = "t"
    else
      l = "f"
    end
    -- writing results
    temp_file:write(string.format("%s %f %s \n", clip_name, confidence, l))

    if i % 200 == 0 then
      print(i, 'finished!')
    end
   -- if if_test==true then
   --    qualitative_analysis(model, opt, x, set_name, split, i, attention_weights)
   -- end
  end
  temp_file:close()
  test_names_file:close()

end

--- evaluation multimodal fusion
local function evaluation_set_performance2(opt, model, visual, motion, true_labels, if_test, set_name, loader, split)
  local total_loss_avg = 0
  local accuracy = 0
  local data_size = true_labels:size(1)
  local batch_size = opt.batch_size
  local temp_idx = 1
  local cc = 1
  for i = 1, data_size do
    local x, true_y
    x1 = visual[i]
    x2 = motion[i]

    if opt.gpuid>= 0 then
      x1 = prepro(x1)
      x2 = prepro(x2)
    end

    if x1:dim() == 3 and x1:size(1) == 1 then
      x1 = x1:view(x1:size(2), x1:size(3))
      x2 = x2:view(x2:size(2), x2:size(3))
    end
    true_y = true_labels[i] + 1

    local temp_loss, predict_label, attention_weights = inference(model, x1, x2, true_y, opt)

    total_loss_avg = temp_loss + total_loss_avg
    if predict_label == true_y then
      accuracy = accuracy + 1
    -- predictions[i] = 1
    end
    if i % 200 == 0 then
      print(i, 'finished!')
    end
    -- analise qualitativa
    --if if_test==true then
    --   qualitative_analysis(model, opt, x, set_name, split, i, attention_weights)
    --end
  end

  total_loss_avg = total_loss_avg / data_size
  accuracy = accuracy / data_size * 100

  return total_loss_avg, accuracy
end

--- input @data_set is a data sequence (table of Tensor) to be evaluated
local function evaluation_set_performance(opt, model, data_sequence, true_labels, if_test, set_name, loader, split)
  local total_loss_avg = 0
  local accuracy = 0
  local correct = 0
  local recall = 0
  local total_pos = 0
  local data_size = true_labels:size(1)
  local batch_size = opt.batch_size
  local temp_idx = 1
  local cc = 1
  local all_attention_weights = {}
  for i = 1, data_size do
    local x, true_y
    x = data_sequence[i]
    if x:dim()==3 then
      x = x:view(x:size(2), x:size(3))
    end
    true_y = true_labels[i] + 1


    if opt.gpuid >= 0 then
      x = prepro(x)
    end
    local temp_loss, predict_label, attention_weights = inference(model, x, true_y, opt)

    all_attention_weights[#all_attention_weights+1] = attention_weights:clone()
    total_loss_avg = temp_loss + total_loss_avg

    if opt.metric == 'accuracy' then
      if predict_label == true_y and opt.metric == 'accuracy' then
        accuracy = accuracy + 1
      end
    end

    if opt.metric == 'recall' then
      if predict_label == 2 and true_y == 2 then
        correct = correct + 1
      end

      if predict_label == 1 and true_y == 2 then
        incorrect = incorrect + 1
      end
    end


    if i % 200 == 0 then
      print(i, 'finished!')
    end
   -- if if_test==true then
   --    qualitative_analysis(model, opt, x, set_name, split, i, attention_weights)
   -- end
  end

  total_loss_avg = total_loss_avg / data_size
  if opt.metric == "accuracy" then
    accuracy = accuracy / data_size * 100
    return total_loss_avg, accuracy
  else
    recall = correct / (correct + incorrect)
    return total_loss_avg, recall
  end

end

--- evaluate the data set
function evaluate_process.evaluate_set(set_name, opt, loader, model, split, test_clips)
  print('start to evaluate the whole ' .. set_name .. ' set...')
  local timer = torch.Timer()
  local time_s = timer:time().real
  if split == -1  then
    split = 0
  end
  local total_loss_avg = nil
  local accuracy = nil
  if opt.mediaeval then
    evaluation_set_mediaeval(opt, model,loader.test_X_visual,loader.test_X_motion, test_clips)
    -- total_loss_avg, accuracy = evaluation_set_performance(opt, model,
    --     loader.train_X_visual,loader.train_T, false, set_name, loader, split)
  elseif set_name == 'train' then
    total_loss_avg, accuracy = evaluation_set_performance(opt, model,
      loader.train_X,loader.train_T, false, set_name, loader)
    --      image_display(model, opt, loader.train_X, 'train')
  elseif set_name == 'validation' then
    total_loss_avg, accuracy = evaluation_set_performance(opt, model,
      loader.validation_X,loader.validation_T, false, set_name, loader)
    --      image_display(model, opt, loader.validation_X, 'validation')
  elseif set_name == 'test' then
    if opt.multimodal then
      total_loss_avg, accuracy = evaluation_set_performance2(opt, model,
        loader.test_X_visual,loader.test_X_motion,loader.valid_y, true, set_name, loader, split)
    else
      total_loss_avg, accuracy = evaluation_set_performance(opt, model,
        loader.test_X_visual,loader.test_T, true, set_name, loader, split)
    end
    --      image_display(model, opt, loader.test_X, 'test')
  else
    error('there is no such set name!')
  end
  local time_e = timer:time().real
  print('total average loss of ' .. set_name .. ' set:', total_loss_avg)
  print('accuracy: ', accuracy)
  print('elapsed time for evaluating the ' .. set_name .. ' set:', time_e - time_s)
  return total_loss_avg, accuracy
end

--- load the data and the trained model from the check point and evaluate the model
function evaluate_process.evaluate_from_scratch(opt, if_train_validation)

  ------------------- create the data loader class ----------
  local loader = data_loader.create(opt)
  local feature_dim = loader.feature_dim
  local do_random_init = true

  ------------------ begin to define the whole model --------------------------
  local model = define_my_model.define_model(opt, loader, true)
  define_my_model.load_model(opt,model, false)
  local if_plot = false
  ------------------- create the data loader class ----------
  print('evaluate the model from scratch...')
  local train_loss, train_accuracy = nil
  local validation_loss, validation_accuracy = nil
  if if_train_validation then
    train_loss, train_accuracy = evaluate_process.evaluate_set('train', opt, loader, model, false)
    validation_loss, validation_accuracy = evaluate_process.evaluate_set('validation', opt, loader, model, false)
  end
  local test_loss, test_accuracy = evaluate_process.evaluate_set('test', opt, loader, model, true)

  local temp_file = io.open(string.format('%s/%s_results_GPU_%d_dropout_%1.2f.txt',
    opt.current_result_dir, opt.opt_method, opt.gpuid, opt.dropout), "a")
  temp_file:write(string.format('similarity measurement results \n'))
  if if_train_validation then
    temp_file:write(string.format('train set loss = %6.8f, train accuracy= %6.8f\n',
      train_loss, train_accuracy ))
    temp_file:write(string.format('validation set loss = %6.8f, validation accuracy = %6.8f\n',
      validation_loss, validation_accuracy ))
  end
  temp_file:write(string.format('test set loss = %6.8f, test accuracy = %6.8f\n',
    test_loss, test_accuracy ))

  if if_train_validation then
    return train_accuracy, validation_accuracy, test_accuracy
  else
    return test_accuracy
  end
end

return evaluate_process

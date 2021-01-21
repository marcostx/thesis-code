
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
local function prepro(opt, x)
 -- if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
    x = x:float():cuda()
 -- end
  return x
end

local function qualitative_analysis( model, opt, x, vid_id, attention_weights)

  local temp_file = io.open(string.format("data/rwf/TAGM_%d_att_weights.csv", vid_id), "w")
  temp_file:write('attention scores, index \n')
  -- image.display(original)

  local attention_w = {}
  for i=1,attention_weights:size(1) do
    tmp = attention_weights[i]
    temp_file:write(string.format("%f, %d \n", tmp, i))
  end
  temp_file:close()

end


--- inference one sample
local function inference(model, x, true_y, opt)

  -- decode the model and parameters
  local attention = model.attention
  local top_net = model.top_net
  local criterion = model.criterion
  local params_flat = model.params_flat
  local x_length = x:size(2)

  -- perform the forward pass for attention model
  local attention_weights, hidden_z_value
  if opt.if_attention == 1 then
    attention_weights, hidden_z_value = attention.forward(x, opt, 'test')
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
    net_output = top_net.forward(x, attention_weights, opt, 'test')
  else
    error('no such top classifier!')
  end
  --compute the loss
  --  local current_loss = criterion:forward(net_output, torch.Tensor({true_y})) -- for batch_size == 1
  local current_loss = criterion:forward(net_output, true_y)
  local _, pred_label = net_output:squeeze():max(1)
  local confidence = net_output[2]

  if opt.if_attention == 0 and opt.top_c == 'lstm' then
    attention_weights:resize(1, opt.top_lstm_size, x_length)
    local max_v = torch.max(attention_weights)
    attention_weights:div(max_v)
  end


  return current_loss, pred_label:squeeze(), confidence, attention_weights
end

-- get the prediction and confidence score for one sample
local function predict(model, x, opt)

  -- decode the model and parameters
  local attention = model.attention
  local top_net = model.top_net
  local criterion = model.criterion
  local params_flat = model.params_flat
  local x_length = x:size(2)

  -- perform the forward pass for attention model
  local attention_weights, hidden_z_value
  if opt.if_attention == 1 then
    attention_weights, hidden_z_value = attention.forward(x, opt, 'test')
  else
    attention_weights = torch.ones(1, x_length)
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
    net_output = top_net.forward(x, attention_weights, opt, 'test')
  else
    error('no such top classifier!')
  end
  --compute the loss
  local _, pred_label = net_output:squeeze():max(1)
  local confidence = net_output[2]

  return confidence,  pred_label:squeeze(), attention_weights
end


--- input @data_set is a data sequence (table of Tensor) to be evaluated
local function evaluation_set_mediaeval(opt, model, data_sequence)
  local l
  local accuracy = 0
  local total_pos = 0
  local data_size = data_sequence:size(1)
  local all_attention_weights = {}
  local predictions = torch.zeros(data_size)
  -- local testPath = 'datasets/test/*.mp4'
  -- local trainPath = 'datasets/train/*.mp4'
  -- local test_cli=ps = ls(testPath)
  -- local train_clips = ls(trainPath)

  -- temp_file = io.open("mediaeval_test_predictions.txt", "w")
  temp_file = io.open("mediaeval_2015_test_predictions_raw.txt", "w")
  local test_names_file = io.open("test_mediaeval_raw.txt", "r")
  local test_clips = {}
  for line in test_names_file:lines() do
    table.insert (test_clips, line);
  end

  for i = 1, data_size do
    local x, true_y
    x = data_sequence[i]
    -- clip_name = string.gsub(test_clips[i], "(.*/)(.*)", "%2")
    -- clip_name = string.sub(clip_name, 1, -5)
    clip_name = test_clips[i]
    if x:dim()==3 then
      x = x:view(x:size(2), x:size(3))
    end
    x = prepro(opt, x)

    local confidence, predict_label, attention_weights = predict(model, x, opt)

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
  local all_attention_weights = {}
  local predictions = torch.zeros(data_size)
  for i = 1, data_size do
    local x, true_y
    x1 = visual[i]
    x2 = motion[i]

    if x1:dim() == 3 and x1:size(1) == 1 then
      x1 = x1:view(x1:size(2), x1:size(3))
      x2 = x2:view(x2:size(2), x2:size(3))
    end
    true_y = true_labels[i] + 1

    if opt.multimodal then
      if opt.early then
        x = torch.cat(x1, x2, 1)
      else
        x = model.att_fusion.forward(x1, x2, opt)
      end
    end

    x = prepro(opt, x)

    local temp_loss, predict_label, attention_weights = inference(model, x, true_y, opt)

    all_attention_weights[#all_attention_weights+1] = attention_weights:clone()
    total_loss_avg = temp_loss + total_loss_avg
    if predict_label == true_y then
      accuracy = accuracy + 1
      predictions[i] = 1
    end
    if i % 200 == 0 then
      print(i, 'finished!')
    end
    -- analise qualitativa
    -- if if_test==true then
    --   qualitative_analysis(model, opt, x, set_name, split, i, attention_weights)
    -- end
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
  local predictions = torch.zeros(data_size)
  for i = 1, data_size do
    local x, true_y
    x = data_sequence[i]
    if x:dim()==3 and x:size(1) == 1 then
      x = x:reshape(x:size(2), x:size(3))
    end
    true_y = true_labels[i] + 1

    x = prepro(opt, x)

   if opt.gpuid >= 0 and opt.opencl == 0 then
      true_y = true_y:float():cuda()
    end
    local temp_loss, predict_label, confidence, attention_weights = inference(model, x, true_y, opt)

    all_attention_weights[#all_attention_weights+1] = attention_weights:clone()
    total_loss_avg = temp_loss + total_loss_avg

    if predict_label == true_y and opt.metric == 'accuracy' then
      accuracy = accuracy + 1
      predictions[i] = 1
    end
    

    if predict_label == 2 and true_y == 2 and opt.metric == 'recall' then
      correct = correct + 1
      predictions[i] = 1
    end

    if true_y == 2 then
      total_pos = total_pos + 1
    end

    if i % 200 == 0 then
      print(i, 'finished!')
    end
    -- if predict_label == true_y then
    --   print(i)
    --   print(true_y)
    --   print(predict_label)
    --   print(confidence)
    -- end
  --  if if_test==true and i == 132 then
  --     print("running qualitative analysis!!")
  --     print(true_y, predict_label)
  --     qualitative_analysis(model, opt, x, i, attention_weights)
  --  end
  end

  total_loss_avg = total_loss_avg / data_size
  if opt.metric == "accuracy" then
    accuracy = accuracy / data_size * 100
    return total_loss_avg, accuracy
  else
    recall = correct / total_pos
    return total_loss_avg, recall
  end

end

--- evaluate the data set
function evaluate_process.evaluate_set(set_name, opt, loader, model, split)
  print('start to evaluate the whole ' .. set_name .. ' set...')
  local timer = torch.Timer()
  local time_s = timer:time().real
  if split == -1  then
    split = 0
  end
  local total_loss_avg = nil
  local accuracy = nil
  if opt.mediaeval then
    evaluation_set_mediaeval(opt, model, loader.valid_X)
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
        loader.test_X_visual,loader.test_X_motion,loader.test_T, true, set_name, loader, split)
    else
      total_loss_avg, accuracy = evaluation_set_performance(opt, model,
        loader.valid_X,loader.valid_y, true, set_name, loader, split)
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
  
  loader.train_X = loader.data.train_X
  loader.valid_X = loader.data.valid_X
  loader.train_y = loader.data.train_y
  loader.valid_y = loader.data.valid_y
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
  local test_loss, test_accuracy = evaluate_process.evaluate_set('test', opt, loader, model)

  --ocal temp_file = io.open(string.format('%s/%s_results_GPU_%d_dropout_%1.2f.txt',
  --  opt.current_result_dir, opt.opt_method, opt.gpuid, opt.dropout), "a")
  -- temp_file:write(string.format('similarity measurement results \n'))
  -- if if_train_validation then
  --  temp_file:write(string.format('train set loss = %6.8f, train accuracy= %6.8f\n',
  --    train_loss, train_accuracy ))
  --  temp_file:write(string.format('validation set loss = %6.8f, validation accuracy = %6.8f\n',
  --    validation_loss, validation_accuracy ))
  -- end
  -- temp_file:write(string.format('test set loss = %6.8f, test accuracy = %6.8f\n',
  --  test_loss, test_accuracy ))

  if if_train_validation then
    return train_accuracy, validation_accuracy, test_accuracy
  else
    return test_accuracy
  end
end

return evaluate_process

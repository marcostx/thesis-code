
require 'image'

local path = require 'pl.path'
require 'util.misc'
local data_loader = require 'util.data_loader'
local model_utils = require 'util.model_utils'
local define_my_model = require 'model.define_my_model'
local table_operation = require 'util/table_operation'

function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

local evaluate_process = {}

--- preprocessing helper function
local function prepro(x)
  --if opt.gpuid >= 0 then -- ship the input arrays to GPU
    x = x:float():cuda()
  --end
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


--- inference one sample
local function inference(model, x, true_y, opt)

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

  -- decode the model and parameters
  local attention = model.attention
  local top_net = model.top_net
  local cnn_encoder = model.cnn_encoder
  local rnn_decoder = model.rnn_decoder
  local convlstm = model.convlstm
  local criterion = model.criterion
  local params_flat = model.params_flat
  local x_length = x:size(4)

  -- perform the forward pass for attention model
  local attention_weights, hidden_z_value
  attention_weights = torch.ones(x_length)
  
  -- perform the forward for the top-net module
  local net_output = nil
  if opt.top_c == 'NN' then
    if opt.if_original_feature == 1 then
      net_output = top_net:forward({x, attention_weights})
    else
      net_output = top_net:forward({hidden_z_value, attention_weights})
    end
  elseif opt.top_c == 'lstm' or opt.top_c == 'rnn' or opt.top_c == 'gru' or opt.top_c == 'TAGM' or opt.top_c == 'tsam' then
    net_output = top_net.forward(x, attention_weights, opt, 'test')
  elseif opt.top_c == 'cnn' then
    net_output = top_net.forward(x, opt, 'test')
  elseif opt.top_c == 'vgg_att' then
    cnn_embeddings,l1_, l2_, l3_, conv_out_, fc1_, fc1_l1_,fc1_l2_,fc1_l3_, g = cnn_encoder.forward(x, opt, 'training')
    attention_weights, hidden_z_value = attention.forward(cnn_embeddings, opt, 'training')
    net_output = rnn_decoder.forward(cnn_embeddings,attention_weights, opt, 'training')
  elseif opt.top_c == 'convlstm' then
      cnn_embeddings = cnn_encoder.forward(x, opt, 'test')
      if opt.if_attention == 1 then
        attention_weights, hidden_z_value = attention.forward(cnn_embeddings, opt, 'test')
      else
        attention_weights = torch.ones(x_length)
      end
      net_output = rnn_decoder.forward(cnn_embeddings,attention_weights, opt, 'test')
  elseif opt.top_c == 'convlstm_v2' then
      net_output = convlstm.forward(x, opt, 'training')
  else
    error('no such top classifier!')
  end
  --compute the loss
  --  local current_loss = criterion:forward(net_output, torch.Tensor({true_y})) -- for batch_size == 1
  local current_loss = criterion:forward(net_output, true_y)
  local _, pred_label = net_output:squeeze():max(1)

  return current_loss, pred_label:squeeze()
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
  elseif opt.top_c == 'lstm' or opt.top_c == 'rnn' or opt.top_c == 'gru' or opt.top_c == 'TAGM' or opt.top_c == 'tsam'  then
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
local function evaluation_set_mediaeval(opt, model, visual, motion, test_clips)
  local l
  local accuracy = 0
  local total_pos = 0
  local data_size = visual:size(1)
  local all_attention_weights = {}
  local predictions = torch.zeros(data_size)
  -- local testPath = 'data/test_fps15/*.npy'
  -- local trainPath = 'data/train/*.npy'
  -- local test_clips = ls(testPath)
  -- local train_clips = ls(trainPath)

  -- temp_file = io.open("mediaeval_test_predictions.txt", "w")
  temp_file = io.open("mediaeval_test_predictions_vgg_hog.txt", "w")

  for i = 1, data_size do
    local x
    
    clip_name = string.gsub(test_clips[i], "(.*/)(.*)", "%2")
    clip_name = string.sub(clip_name, 1, -5)
    if opt.multimodal then
      x1 = visual[i]
      x2 = motion[i]

      if opt.gpuid>= 0 then
        x1 = prepro(x1)
        x2 = prepro(x2)
      end

      if x1:dim() == 3 then
        x1 = x1:view(x1:size(2), x1:size(3))
        x2 = x2:view(x2:size(2), x2:size(3))
      end
    else
      x = visual[i]
      if opt.gpuid>= 0 then
        x = prepro(x)
      end
      if x:dim() == 3 then
        x = x:view(x:size(2), x:size(3))
      end
    end

    if opt.multimodal then
      x = model.att_fusion.forward(x1, x2, opt)
    end

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

    -- x1 = prepro(x1)
    -- x2 = prepro(x2)


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

    -- = prepro(opt, x)

    local temp_loss, predict_label, attention_weights = inference(model, x, true_y, opt)

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
    local temp_loss, predict_label = inference(model, x, true_y, opt)

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
        loader.test_X_visual,loader.test_X_motion,loader.test_T, true, set_name, loader, split)
    else
      total_loss_avg, accuracy = evaluation_set_performance(opt, model,
        loader.testX,loader.testY, true, set_name, loader, split)
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


return evaluate_process

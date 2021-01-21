

require 'nn'
require 'nngraph'
local path = require 'pl.path'

-- local library
local RNN = require 'model.my_RNN'
local LSTM = require 'model.my_LSTM'
local attention = require 'model.attention_v2'
local model_utils = require 'util.model_utils'
local data_loader = require 'util.data_loader'
require 'util.misc'
local Top_NN = require 'model.Top_NN_Classifier'
local Top_LSTM = require 'model.Top_LSTM_Classifier'
local Top_RNN = require 'model.Top_RNN_Classifier'
local TAGM = require 'model.TAGM_v2'
local TSAM = require 'model.TSAM'
local CNN = require 'model.Top_CNN'
local vgg_att = require 'model.VGG_ATT'
local encCNN = require 'model.EncoderCNN'
local convlstm = require 'model.ConvLSTM'
local Multimodal_Fusion = require 'model.Multimodal_Fusion'

local my_model = {}

--- save the current trained best model
-- for the continution of training or for the test
function my_model.save_model(opt, model)
  local savefile = opt.savefile
  print('saving checkpoint to ' .. savefile)
  opt.savefile = savefile
  -- to save the space, we only save the parameter values in the model
  local checkpoint = {
    params_flat = model.params_flat,
    learning_rate = opt.learning_rate
  }
  torch.save(savefile, checkpoint)
end

function my_model.load_model(opt, model, if_learning_rate)

  local savefile = opt.savefile
  local checkpoint = torch.load(savefile)
  model.params_flat:copy(checkpoint.params_flat)
  if if_learning_rate then
    opt.learning_rate = checkpoint.learning_rate
  end
  return model, opt
end

function my_model.define_model(opt, loader, if_evaluate_from_scratch)

  local model = {}


  ------------------ for attention model --------------------------
  if opt.if_attention == 1 then
    print('creating attention model with ' .. opt.attention_num_layers .. ' layers')
    attention.model(loader, opt)
  end

  ------------ define the top net module to connect two rnn modules---------
  print('creating the top net for classification...')
  local top_net = nil
  local cnn_encoder = nil
  local rnn_decoder = nil
  local cLSTM = nil
  if opt.top_c == 'NN' then
    top_net = Top_NN.net(loader, opt, loader.class_size)
    local top_net_params_flat, top_net_grad_params_flat = top_net:getParameters()
    print('number of parameters in the top_net model: ' .. top_net_params_flat:nElement())
    model.top_net_params_size = top_net_params_flat:nElement()
  elseif opt.top_c == 'lstm' or opt.top_c == 'rnn' or opt.top_c == 'gru' then
    top_net = Top_LSTM
    Top_LSTM.net(loader, opt)
    model.top_net_params_size = top_net.params_size
  elseif opt.top_c == 'TAGM' then
    top_net = TAGM
    TAGM.model(loader, opt)
    model.top_net_params_size = top_net.params_size
  elseif opt.top_c == 'tsam' then
    top_net = TSAM
    TSAM.net(loader, opt)
    model.top_net_params_size = top_net.params_size
  elseif opt.top_c == 'cnn' then
    top_net = CNN
    CNN.net(loader, opt)
    model.top_net_params_size = top_net.params_size
  elseif opt.top_c == 'convlstm' then
    cnn_encoder = encCNN
    rnn_decoder = TAGM

    cnn_encoder.net(loader, opt)
    rnn_decoder.model(loader, opt)
    model.top_net_params_size = cnn_encoder.params_size + rnn_decoder.params_size
  elseif opt.top_c == 'convlstm_v2' then
    cLSTM = convlstm

    cLSTM.net(loader, opt)
    model.top_net_params_size = cLSTM.params_size
  elseif opt.top_c == 'vgg_att' then
    cnn_encoder = vgg_att
    rnn_decoder = TAGM

    cnn_encoder.net(loader, opt)
    rnn_decoder.model(loader, opt)
    model.top_net_params_size = cnn_encoder.params_size + rnn_decoder.params_size
  else
    error('no such top classifier!')
  end


  --------- define the criterion (loss function) ---------------
  local criterion = nn.ClassNLLCriterion()
  model.criterion = criterion
  -- ship the model to the GPU if desired
  if opt.gpuid >= 0 then criterion:cuda() end

  -- get the flat parameters from all modules
  local params_flat, grad_params_flat
  if opt.top_c == 'NN' then
    if opt.if_attention == 1 then
      params_flat, grad_params_flat = model_utils.combine_all_parameters(attention.rnn, attention.birnn, attention.weight_net, top_net)
    else
      params_flat, grad_params_flat = model_utils.combine_all_parameters(top_net)
    end
  elseif opt.top_c == 'lstm' or opt.top_c == 'rnn' or opt.top_c == 'gru' then
    if opt.if_attention == 1 then
      if opt.top_bidirection == 0 then
        params_flat, grad_params_flat = model_utils.combine_all_parameters(attention.rnn, attention.birnn,
          attention.weight_net, top_net.mul_net, top_net.lstm, top_net.top_c)
      else
        params_flat, grad_params_flat = model_utils.combine_all_parameters(attention.rnn, attention.birnn,
          attention.weight_net, top_net.mul_net, top_net.lstm, top_net.bilstm, top_net.top_c)
      end
    else
      if opt.top_bidirection == 0 then
        params_flat, grad_params_flat = model_utils.combine_all_parameters(top_net.mul_net, top_net.rnn, top_net.top_c)
      else
        params_flat, grad_params_flat = model_utils.combine_all_parameters(top_net.mul_net, top_net.rnn, top_net.birnn, top_net.top_c)
      end
    end
  elseif opt.top_c == 'tsam' then
    if opt.if_attention == 1 then
      if opt.top_bidirection == 0 then
        params_flat, grad_params_flat = model_utils.combine_all_parameters(attention.rnn, attention.birnn,
          attention.weight_net, top_net.mul_net, top_net.rnn, top_net.top_c, top_net.pre_m)
      else
        params_flat, grad_params_flat = model_utils.combine_all_parameters(attention.rnn, attention.birnn,
          attention.weight_net, top_net.mul_net, top_net.rnn, top_net.birnn, top_net.top_c, top_net.pre_m)
      end
    else
      if opt.top_bidirection == 0 then
        params_flat, grad_params_flat = model_utils.combine_all_parameters(top_net.mul_net, top_net.rnn, top_net.top_c)
      else
        params_flat, grad_params_flat = model_utils.combine_all_parameters(top_net.mul_net, top_net.rnn, top_net.birnn, top_net.top_c)
      end
    end
  elseif opt.top_c == 'cnn' then
    params_flat, grad_params_flat = model_utils.combine_all_parameters(top_net.cnn)
  elseif opt.top_c == 'vgg_att' then
    params_flat, grad_params_flat = model_utils.combine_all_parameters(cnn_encoder.l1 ,cnn_encoder.l2 ,cnn_encoder.l3 ,
    cnn_encoder.conv_out ,cnn_encoder.fc1,cnn_encoder.fc1_l1 ,cnn_encoder.fc1_l2 ,cnn_encoder.fc1_l3,attention.rnn, attention.birnn,
    attention.weight_net, rnn_decoder.pre_m, rnn_decoder.lstm, rnn_decoder.top_c )
    -- params_flat, grad_params_flat = model_utils.combine_all_parameters(top_net.top_c )
   elseif opt.top_c == 'TAGM' then
    if opt.if_attention == 1 then
      if opt.top_bidirection == 0 then
        params_flat, grad_params_flat = model_utils.combine_all_parameters(attention.rnn1,attention.birnn1,attention.rnn2,attention.birnn2,
          attention.weight_net1,attention.weight_net2, top_net.pre_m, top_net.lstm1, top_net.lstm2, top_net.top_c1, top_net.top_c2)

        print('number of parameters in the whole model: ' .. params_flat:nElement())
      else
        params_flat, grad_params_flat = model_utils.combine_all_parameters(attention.rnn, attention.birnn,
          attention.weight_net, top_net.pre_m, top_net.lstm, top_net.bilstm, top_net.top_c)
      end
    else
      if opt.top_bidirection == 0 then
        params_flat, grad_params_flat = model_utils.combine_all_parameters(top_net.pre_m, top_net.lstm, top_net.top_c)
      else
        params_flat, grad_params_flat = model_utils.combine_all_parameters(top_net.pre_m, top_net.lstm, top_net.bilstm, top_net.top_c)
      end
    end
  elseif opt.top_c == 'convlstm' then
    if opt.if_attention == 1 then
      params_flat, grad_params_flat = model_utils.combine_all_parameters(attention.rnn, attention.birnn,
            attention.weight_net, rnn_decoder.pre_m, rnn_decoder.lstm, rnn_decoder.top_c,
                                      cnn_encoder.cnn)
    else
      params_flat, grad_params_flat = model_utils.combine_all_parameters(rnn_decoder.pre_m, rnn_decoder.lstm, rnn_decoder.top_c,
                                      cnn_encoder.cnn)
    end
  elseif opt.top_c == 'convlstm_v2' then
  params_flat, grad_params_flat = model_utils.combine_all_parameters(cLSTM.convlstm)
  else
    error('no such top classifier')
  end
  -- clone the rnn and birnn
  if opt.if_attention == 1 then
    print("cloning attention net")
    attention.clone_model(loader)
  end

  if opt.top_c == 'convlstm' or opt.top_c == 'TAGM' or opt.top_c == 'tsam' or opt.top_c == 'vgg_att' then
    if opt.top_c == 'convlstm' or opt.top_c == 'vgg_att' then
      rnn_decoder.clone_model(loader, opt)
    else
      top_net.clone_model(loader, opt)
    end
  end
  model.attention = attention
  model.params_flat = params_flat
  model.grad_params_flat = grad_params_flat
  print('number of parameters in the whole model: ' .. params_flat:nElement())

  if opt.if_attention == 1 then
    model.attention_params_size = attention.params_size
  end
  model.params_size = params_flat:nElement()
  model.top_net = top_net
  if opt.top_c == 'convlstm' or opt.top_c == 'vgg_att' then
    model.cnn_encoder = cnn_encoder
    model.rnn_decoder = rnn_decoder
  end

  if opt.top_c == 'convlstm_v2' then
    model.convlstm = cLSTM
  end

  if opt.if_init_from_check_point or if_evaluate_from_scratch then
    if path.exists(savefile) then
      print('Init the model from the check point saved before...\n')
      model = my_model.load_model(opt, model, true)
    else
      error('error: there is no trained model saved before in such experimental setup.')
    end
  else
    if opt.top_c ~= 'NN' then
      if opt.if_attention == 1 then
        attention.init_lstm(opt)
      end
    end
  end

  -- pre-allocate the memory for the temporary variable used in the training phase
  local params_grad_all_batches = torch.zeros(grad_params_flat:nElement())
  if opt.gpuid >= 0 then
    params_grad_all_batches = params_grad_all_batches:float():cuda()
  end
  model.params_grad_all_batches = params_grad_all_batches

  return model, opt
end

return my_model

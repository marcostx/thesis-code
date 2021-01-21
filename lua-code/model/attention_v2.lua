  
----
--given a time series, the attention model output a weight or a weighted vector for each time step of time series

require 'nn'
local RNN = require 'model.my_RNN'
local LSTM = require 'model.my_LSTM'
local model_utils = require 'util.model_utils'
--require 'model.myMapTable'
require 'util/misc'

local Attention = {}

local function weight_model(loader, opt)
  local input = {}
  local model = nn.Sequential()
  local map = nn.MapTable()
  local linear = nn.Linear(2*opt.att_rnn_size, 1)
  map:add(linear)
  model:add(map)
  model:add(nn.JoinTable(1, 1))
  model:add(nn.MulConstant(opt.attention_sig_w))
  model:add(nn.Sigmoid())
  return model

end

function Attention.model(loader, opt)

  local rnn_model1 = nil
  local rnn_model2 = nil
  local birnn_model1 = nil 
  local birnn_model2 = nil 

  rnn_model1 = RNN.rnn(loader.visual_feature_dim, opt.att_rnn_size, opt.attention_num_layers, opt.dropout)
  birnn_model1 = RNN.rnn(loader.visual_feature_dim, opt.att_rnn_size, opt.attention_num_layers, opt.dropout)

  rnn_model2 = RNN.rnn(loader.motion_feature_dim, opt.att_rnn_size, opt.attention_num_layers, opt.dropout)
  birnn_model2 = RNN.rnn(loader.motion_feature_dim, opt.att_rnn_size, opt.attention_num_layers, opt.dropout)

  -- the initial state of the cell/hidden states
  local rnn_init_state1 = {}
  local rnn_init_state2 = {}
  for L=1,opt.attention_num_layers do
    local h_init = torch.zeros(1, opt.att_rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(rnn_init_state1, h_init:clone())
    table.insert(rnn_init_state2, h_init:clone())
  end
  Attention.rnn_init_state1 = rnn_init_state1
  Attention.rnn_init_state2 = rnn_init_state2
  local weight_net1 = weight_model(loader, opt)
  local weight_net2 = weight_model(loader, opt)
  Attention.weight_net1 = weight_net1
  Attention.weight_net2 = weight_net2
  Attention.rnn1 = rnn_model1
  Attention.birnn1 = birnn_model1
  Attention.rnn2 = rnn_model2
  Attention.birnn2 = birnn_model2
  -- ship the model to the GPU if desired
  if opt.gpuid >= 0 then rnn_model1:cuda(); birnn_model1:cuda(); weight_net1:cuda();rnn_model2:cuda(); birnn_model2:cuda(); weight_net2:cuda(); end
  local rnn_params_flat1, rnn_grad_params_flat1 = rnn_model1:getParameters()
  local rnn_params_flat2, rnn_grad_params_flat2 = rnn_model2:getParameters()
  Attention.params_size = rnn_params_flat1:nElement()*2 + rnn_params_flat2:nElement()*2 + 2*weight_net1:getParameters():nElement()
  print('number of parameters in the attention model: ' .. Attention.params_size)
end

function Attention.clone_model(loader)
  -- make a bunch of clones for input time series after flattening, sharing the same memory
  -- note that: it is only performed once for the reason of efficiency,
  -- hence we clone the max length of times series in the data set for each rnn time series
  print('cloning rnn')
  local clones_rnn1 = model_utils.clone_many_times(Attention.rnn1, loader.max_time_series_length)
  print('cloning ' .. loader.max_time_series_length ..  ' rnns for each time series finished! ')
  print('cloning bidirectional rnn')
  local clones_birnn1 = model_utils.clone_many_times(Attention.birnn1, loader.max_time_series_length)
  print('cloning ' .. loader.max_time_series_length ..  ' birnns for each time series finished! ')


  print('cloning rnn')
  local clones_rnn2 = model_utils.clone_many_times(Attention.rnn2, loader.max_time_series_length)
  print('cloning ' .. loader.max_time_series_length ..  ' rnns for each time series finished! ')
  print('cloning bidirectional rnn')
  local clones_birnn2 = model_utils.clone_many_times(Attention.birnn2, loader.max_time_series_length)
  print('cloning ' .. loader.max_time_series_length ..  ' birnns for each time series finished! ')


  Attention.clones_rnn1 = clones_rnn1
  Attention.clones_rnn2 = clones_rnn2
  Attention.clones_birnn1 = clones_birnn1
  Attention.clones_birnn2 = clones_birnn2
end

function Attention.init_lstm(opt)
  -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
  if opt.att_model == 'lstm' then
    for layer_idx = 1, opt.attention_num_layers do
      for _,node in ipairs(Attention.rnn1.forwardnodes) do
        if node.data.annotations.name == "i2h_" .. layer_idx then
          print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
          -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
          node.data.module.bias[{{opt.att_rnn_size+1, 2*opt.att_rnn_size}}]:fill(1.0)
        end
      end
    end
  end
  for layer_idx = 1, opt.attention_num_layers do
    for _,node in ipairs(Attention.birnn1.forwardnodes) do
      if node.data.annotations.name == "i2h_" .. layer_idx then
        print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
        -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
        node.data.module.bias[{{opt.att_rnn_size+1, 2*opt.att_rnn_size}}]:fill(1.0)
      end
    end
  end
end

function Attention.forward(x, opt, flag, modality)
  local weights
  local tmp_hidden_z_value = {}  -- the value of rnn1 hidden unit in the last time step
  local tmp_hidden_z_value2 = {}  -- the value of rnn1 hidden unit in the last time step
  local hidden_z_value = torch.Tensor(x:size(2),opt.att_rnn_size*2)

  if modality == 1 then
    local rnn_init_state = Attention.rnn_init_state1
    local x_length = x:size(2)
    local clones_rnn = Attention.clones_rnn1
    local clones_birnn = Attention.clones_birnn1
    local weight_net = Attention.weight_net1

    local init_state_global = clone_list(rnn_init_state)

    -- perform forward for forward rnn
    local rnn_input = x
    local rnn_state = {[0] = init_state_global}
    -- we don't set the opt.seq_length, instead, we use the current length of the time series
    for t=1,x_length do
      if flag == 'test' then
        clones_rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
      else
        clones_rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      end
      local lst = clones_rnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#rnn_init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      -- last element is the output of the current time step: the hidden value after dropout
      tmp_hidden_z_value[t] = lst[#lst]
    end
    Attention.rnn_state1 = rnn_state

    -- perform the forward pass for birnn: in the other direction
    local birnn_state, bihidden_z_value
    birnn_state = {[x_length+1] = init_state_global}
    bihidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step
    -- we don't set the opt.seq_length, instead, we use the current length of the time series
    for t=x_length, 1, -1 do
      if flag == 'test' then
        clones_birnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
      else
        clones_birnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      end
      local lst = clones_birnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), unpack(birnn_state[t+1])}
      birnn_state[t] = {}
      for i=1,#rnn_init_state do table.insert(birnn_state[t], lst[i]) end -- extract the state, without output
      -- last element is the output of the current time step: the hidden value after dropout
      
      tmp_hidden_z_value2[t] = torch.cat(tmp_hidden_z_value[t], lst[#lst], 1)
      -- hidden_z_value[t] = torch.cat(tmp_hidden_z_value[t], lst[#lst], 1)

    end
    Attention.birnn_state1 = birnn_state
    weights = weight_net:forward(tmp_hidden_z_value2)
  else
    local rnn_init_state = Attention.rnn_init_state2
    local x_length = x:size(2)
    local clones_rnn = Attention.clones_rnn2
    local clones_birnn = Attention.clones_birnn2
    local weight_net = Attention.weight_net2

    local init_state_global = clone_list(rnn_init_state)

    -- perform forward for forward rnn
    local rnn_input = x
    local rnn_state = {[0] = init_state_global}
    -- we don't set the opt.seq_length, instead, we use the current length of the time series
    for t=1,x_length do
      if flag == 'test' then
        clones_rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
      else
        clones_rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      end
      local lst = clones_rnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#rnn_init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      -- last element is the output of the current time step: the hidden value after dropout
      tmp_hidden_z_value[t] = lst[#lst]
    end
    Attention.rnn_state2 = rnn_state

    -- perform the forward pass for birnn: in the other direction
    local birnn_state, bihidden_z_value
    birnn_state = {[x_length+1] = init_state_global}
    bihidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step
    -- we don't set the opt.seq_length, instead, we use the current length of the time series
    for t=x_length, 1, -1 do
      if flag == 'test' then
        clones_birnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
      else
        clones_birnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      end
      local lst = clones_birnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), unpack(birnn_state[t+1])}
      birnn_state[t] = {}
      for i=1,#rnn_init_state do table.insert(birnn_state[t], lst[i]) end -- extract the state, without output
      -- last element is the output of the current time step: the hidden value after dropout
      tmp_hidden_z_value2[t] = torch.cat(tmp_hidden_z_value[t], lst[#lst], 1)
      -- hidden_z_value[t] = torch.cat(tmp_hidden_z_value[t], lst[#lst], 1)
    end
    Attention.birnn_state2 = birnn_state
    weights = weight_net:forward(tmp_hidden_z_value2)
  end

  return weights, tmp_hidden_z_value2
end

function Attention.backward(opt, hidden_z_value, gradOut, x, modality)

  if modality == 1 then
    local rnn_init_state = Attention.rnn_init_state1
    local clones_rnn = Attention.clones_rnn1
    local clones_birnn = Attention.clones_birnn1
    local weight_net = Attention.weight_net1
    local rnn_state = Attention.rnn_state1
    local birnn_state = Attention.birnn_state1

    local grad_weights = weight_net:backward(hidden_z_value, gradOut)

    local x_length = x:size(2)
    local rnn_input = x
    -- backward for rnn and birnn
    local drnn_state = {[x_length] = clone_list(rnn_init_state, true)} -- true also zeros the clones
    -- perform back propagation through time (BPTT)
    for t = x_length,1,-1 do
      local doutput_t
      doutput_t = grad_weights[t]:sub(1, opt.att_rnn_size)
      table.insert(drnn_state[t], doutput_t)
      local dlst = clones_rnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), unpack(rnn_state[t-1])}, drnn_state[t])
      drnn_state[t-1] = {}
      for k,v in pairs(dlst) do
        if k ~= 1 then -- k == 1 is gradient on x, which we dont need
          drnn_state[t-1][k-1] = v
        end
      end
    end
    -- backward for birnn
    local bidrnn_state = {[1] = clone_list(rnn_init_state, true)} -- true also zeros the clones
    -- perform back propagation through time (BPTT)
    for t = 1, x_length do
      local doutput_t = grad_weights[t]:sub(opt.att_rnn_size+1, -1)
      table.insert(bidrnn_state[t], doutput_t)
      local dlst = clones_birnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), unpack(birnn_state[t+1])}, bidrnn_state[t])
      bidrnn_state[t+1] = {}
      for k,v in pairs(dlst) do
        if k ~= 1 then -- k == 1 is gradient on x, which we dont need
          bidrnn_state[t+1][k-1] = v
        end
      end
    end
  else
    local rnn_init_state = Attention.rnn_init_state2
    local clones_rnn = Attention.clones_rnn2
    local clones_birnn = Attention.clones_birnn2
    local weight_net = Attention.weight_net2
    local rnn_state = Attention.rnn_state2
    local birnn_state = Attention.birnn_state2
    local grad_weights = weight_net:backward(hidden_z_value, gradOut)

    local x_length = x:size(2)
    local rnn_input = x
    -- backward for rnn and birnn
    local drnn_state = {[x_length] = clone_list(rnn_init_state, true)} -- true also zeros the clones
    -- perform back propagation through time (BPTT)
    for t = x_length,1,-1 do
      local doutput_t
      doutput_t = grad_weights[t]:sub(1, opt.att_rnn_size)
      table.insert(drnn_state[t], doutput_t)
      local dlst = clones_rnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), unpack(rnn_state[t-1])}, drnn_state[t])
      drnn_state[t-1] = {}
      for k,v in pairs(dlst) do
        if k ~= 1 then -- k == 1 is gradient on x, which we dont need
          drnn_state[t-1][k-1] = v
        end
      end
    end
    -- backward for birnn
    local bidrnn_state = {[1] = clone_list(rnn_init_state, true)} -- true also zeros the clones
    -- perform back propagation through time (BPTT)
    for t = 1, x_length do
      local doutput_t = grad_weights[t]:sub(opt.att_rnn_size+1, -1)
      table.insert(bidrnn_state[t], doutput_t)
      local dlst = clones_birnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), unpack(birnn_state[t+1])}, bidrnn_state[t])
      bidrnn_state[t+1] = {}
      for k,v in pairs(dlst) do
        if k ~= 1 then -- k == 1 is gradient on x, which we dont need
          bidrnn_state[t+1][k-1] = v
        end
      end
    end
  end

end

return Attention

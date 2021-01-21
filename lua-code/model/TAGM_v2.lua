
local model_utils = require 'util/model_utils'

local TAGM_v2 = {}

local function TAGM_model(input_size, rnn_size, n, dropout)
  dropout = dropout or 0

  -- there will be n+2 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- attention_weight == input_gate == 1-forget_gate
  for L = 1,n do
    -- since we don't have output gate, hence we prev_c = prev_h
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  local in_gate = inputs[2]
  local forget_gate = nn.AddConstant(1.0)(nn.MulConstant(-1.0)(in_gate))
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L+2]
    -- the input to this layer
    if L == 1 then
      --      x = OneHot(input_size)(inputs[1])
      x = inputs[1]
      input_size_L = input_size
    else
      x = outputs[L-1]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h):annotate{name='h2h_'..L}
    local in_transform = nn.ReLU()(nn.CAddTable()({i2h, h2h}))
    -- decode the gates

    -- perform the LSTM update
    local next_h           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_h}),
      nn.CMulTable()({in_gate, in_transform})
    })
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  local hidden_v = nn.Dropout(dropout)(top_h)
  --- for the normal LSTM
  --  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  --  local logsoft = nn.LogSoftMax()(proj)
  --  table.insert(outputs, logsoft)
  --- in our case, we only use the last time step, hence we don't perform classification inside the LSTM model
  table.insert(outputs, hidden_v)
  return nn.gModule(inputs, outputs)
end

function TAGM_v2.model(loader, opt)

  local pre_m = nn.Replicate(opt.top_lstm_size, 1)
  TAGM_v2.pre_m = pre_m
  local class_n = loader.class_size
  local top_lstm_variant1
  local top_lstm_variant2
  
  top_lstm_variant1 = TAGM_model(loader.visual_feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
  top_lstm_variant2 = TAGM_model(loader.motion_feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
  -- if opt.top_bidirection == 1 then
  --   top_bilstm_variant1 = TAGM_model(loader.feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
  -- end

  -- the initial state of the cell/hidden states
  local rnn_init_state1 = {}
  local rnn_init_state2 = {}
  for L=1,opt.top_num_layers do
    local h_init = torch.zeros(1, opt.top_lstm_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(rnn_init_state1, h_init:clone())
    table.insert(rnn_init_state2, h_init:clone())
  end
  TAGM_v2.rnn_init_state1 = rnn_init_state1
  TAGM_v2.rnn_init_state2 = rnn_init_state2
  local rnn_params_flat1, rnn_grad_params_flat1 = top_lstm_variant1:getParameters()
  local rnn_params_flat2, rnn_grad_params_flat2 = top_lstm_variant2:getParameters()

  if opt.gpuid >= 0 then top_lstm_variant1:cuda(); top_lstm_variant2:cuda()  end
  -- if opt.gpuid >= 0 and opt.top_bidirection == 1 then top_bilstm_variant:cuda()  end
  TAGM_v2.lstm1 = top_lstm_variant1
  TAGM_v2.lstm2 = top_lstm_variant2

  local top_c1 = nn.Sequential()
  top_c1:add(nn.Linear(opt.top_lstm_size, class_n))
  top_c1:add(nn.LogSoftMax())

  local top_c2 = nn.Sequential()
  top_c2:add(nn.Linear(opt.top_lstm_size, class_n))
  top_c2:add(nn.LogSoftMax())

  TAGM_v2.top_c1 = top_c1
  TAGM_v2.top_c2 = top_c2
  if opt.gpuid >= 0 then
    TAGM_v2.top_c1 = TAGM_v2.top_c1:cuda()
    TAGM_v2.top_c2 = TAGM_v2.top_c2:cuda()
    TAGM_v2.lstm1  = TAGM_v2.lstm1:cuda()
    TAGM_v2.lstm2  = TAGM_v2.lstm2:cuda()
    TAGM_v2.pre_m = TAGM_v2.pre_m:cuda()
  end

  TAGM_v2.params_size = rnn_params_flat1:nElement() + rnn_params_flat1:nElement() + top_c1:getParameters():nElement() + top_c2:getParameters():nElement()
  
  print('number of parameters in the top lstm model: ' .. TAGM_v2.params_size)
end

function TAGM_v2.clone_model(loader, opt)
  print('cloning rnn')
  local clones_rnn1 = model_utils.clone_many_times(TAGM_v2.lstm1, loader.max_time_series_length)
  print('cloning ' .. loader.max_time_series_length ..  ' rnns for each visual time series  finished! ')
  TAGM_v2.clones_rnn1 = clones_rnn1

  local clones_rnn2 = model_utils.clone_many_times(TAGM_v2.lstm2, loader.max_time_series_length)
  print('cloning ' .. loader.max_time_series_length ..  ' rnns for each motion time series  finished! ')
  TAGM_v2.clones_rnn2 = clones_rnn2
  -- if opt.top_bidirection == 1 then
  --   print('cloning bidirectional rnn')
  --   local clones_birnn = model_utils.clone_many_times(TAGM_v2.bilstm, loader.max_time_series_length)
  --   print('cloning ' .. loader.max_time_series_length ..  ' birnns for each time series finished! ')
  --   TAGM_v2.clones_birnn = clones_birnn
  -- end
end

function TAGM_v2.init_params(opt)

end

function TAGM_v2.forward(x, attention_weight, opt, flag, modality)
  local hidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step
  local output_v = nil
  local pre_out = TAGM_v2.pre_m:forward(attention_weight)
  if modality == 1 then
    local pre_m = TAGM_v2.pre_m
    local lstm = TAGM_v2.lstm1
    local rnn_init_state = TAGM_v2.rnn_init_state1
    local clones_rnn = TAGM_v2.clones_rnn1
    local clones_birnn = TAGM_v2.clones_birnn1
    
    if opt.gpuid >= 0 then
      attention_weight = attention_weight:cuda()
    end
    -- forward of lstm
    local x_length = x:size(2)
    local init_state_global = clone_list(rnn_init_state)
    local rnn_input = x
    local rnn_state = {[0] = init_state_global}
    -- we don't set the opt.seq_length, instead, we use the current length of the time series
    for t=1,x_length do
      if flag == 'test' then
        clones_rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
      else
        clones_rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      end
      local lst = clones_rnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#rnn_init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      -- last element is the output of the current time step: the hidden value after dropout
      if t == x_length then
        hidden_z_value = lst[#lst]
      end
    end
    TAGM_v2.rnn_state1 = rnn_state

    -- forward of bilstm
    if opt.top_bidirection == 1 then
      local birnn_state = {[x_length+1] = init_state_global}
      local bihidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step
      -- we don't set the opt.seq_length, instead, we use the current length of the time series
      for t=x_length, 1, -1 do
        if flag == 'test' then
          clones_birnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
        else
          clones_birnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        end
        local lst = clones_birnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), unpack(birnn_state[t+1])}
        birnn_state[t] = {}
        for i=1,#rnn_init_state do table.insert(birnn_state[t], lst[i]) end -- extract the state, without output
        -- last element is the output of the current time step: the hidden value after dropout
        if t== 1 then
          bihidden_z_value = lst[#lst]
        end
      end
      TAGM_v2.birnn_state1 = birnn_state
      -- concatenate the output of forward and backward LSTM
      hidden_z_value = torch.cat(hidden_z_value, bihidden_z_value, 1)
    end
    output_v = TAGM_v2.top_c1:forward(hidden_z_value)
    TAGM_v2.hidden_z_value1 = hidden_z_value
  else
    local pre_m = TAGM_v2.pre_m
    local lstm = TAGM_v2.lstm2
    local rnn_init_state = TAGM_v2.rnn_init_state2
    local clones_rnn = TAGM_v2.clones_rnn2
    local clones_birnn = TAGM_v2.clones_birnn2
    
    if opt.gpuid >= 0 then
      attention_weight = attention_weight:cuda()
    end
    -- forward of lstm
    local x_length = x:size(2)
    local init_state_global = clone_list(rnn_init_state)
    local rnn_input = x
    local rnn_state = {[0] = init_state_global}
    -- we don't set the opt.seq_length, instead, we use the current length of the time series
    for t=1,x_length do
      if flag == 'test' then
        clones_rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
      else
        clones_rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      end
      local lst = clones_rnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#rnn_init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      -- last element is the output of the current time step: the hidden value after dropout
      if t== x_length then
        hidden_z_value = lst[#lst]
      end
    end
    TAGM_v2.rnn_state2 = rnn_state

    -- forward of bilstm
    if opt.top_bidirection == 1 then
      local birnn_state = {[x_length+1] = init_state_global}
      local bihidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step
      -- we don't set the opt.seq_length, instead, we use the current length of the time series
      for t=x_length, 1, -1 do
        if flag == 'test' then
          clones_birnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
        else
          clones_birnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        end
        local lst = clones_birnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), unpack(birnn_state[t+1])}
        birnn_state[t] = {}
        for i=1,#rnn_init_state do table.insert(birnn_state[t], lst[i]) end -- extract the state, without output
        -- last element is the output of the current time step: the hidden value after dropout
        if t== 1 then
          bihidden_z_value = lst[#lst]
        end
      end
      TAGM_v2.birnn_state2 = birnn_state
      -- concatenate the output of forward and backward LSTM
      hidden_z_value = torch.cat(hidden_z_value, bihidden_z_value, 1)
    end
    output_v = TAGM_v2.top_c2:forward(hidden_z_value)
    TAGM_v2.hidden_z_value2 = hidden_z_value
  end
  TAGM_v2.pre_out = pre_out
  

  return output_v

end

function TAGM_v2.backward(x, attention_weight, opt, gradout, loader, modality)
  local grad_mul_net = nil
  if modality == 1 then
    local pre_m = TAGM_v2.pre_m
    local lstm = TAGM_v2.lstm1
    local rnn_init_state = TAGM_v2.rnn_init_state1
    local clones_rnn = TAGM_v2.clones_rnn1
    local clones_birnn = TAGM_v2.clones_birnn1
    local rnn_state = TAGM_v2.rnn_state1
    local birnn_state = TAGM_v2.birnn_state1
    local x_length = x:size(2)
    local pre_out = nil
    local drnn_pre = nil

    if opt.gpuid >= 0 then
     attention_weight = attention_weight:cuda()
     gradout = gradout:cuda()
    end
    local top_c1 = TAGM_v2.top_c1
    local hidden_z_value = TAGM_v2.hidden_z_value1
    local bidrnn_pre
    if opt.gpuid >= 0 then
      drnn_pre = torch.CudaTensor(opt.top_lstm_size, x_length):zero()
    else
       drnn_pre = torch.DoubleTensor(opt.top_lstm_size, x_length):zero()
    end
    if opt.top_bidirection == 1 and opt.gpuid>=0 then
      bidrnn_pre = torch.CudaTensor(opt.top_lstm_size, x_length):zero()
    elseif opt.top_bidirection == 1 and opt.gpuid==-1 then
      bidrnn_pre = torch.DoubleTensor(opt.top_lstm_size, x_length):zero()
    end
    local rnn_input = x
    if opt.gpuid >= 0 then
      pre_out = TAGM_v2.pre_out:cuda()
    else
      pre_out = TAGM_v2.pre_out
    end
    
    local top_c_grad = top_c1:backward(hidden_z_value, gradout)
    local grad_net1, grad_net2
    if opt.top_bidirection == 1 then
      grad_net1 = top_c_grad:sub(1, opt.top_lstm_size)
      grad_net2 = top_c_grad:sub(opt.top_lstm_size+1, -1)
    else
      grad_net1 = top_c_grad
    end
    local dzeros = torch.zeros(opt.top_lstm_size)
    if opt.gpuid >= 0 then
      dzeros = dzeros:cuda()
    end
    -- backward for rnn and birnn
    local drnn_state = {[x_length] = clone_list(rnn_init_state, true)} -- true also zeros the clones
    -- perform back propagation through time (BPTT)
    for t = x_length,1,-1 do
      local doutput_t
      if t == x_length then
        doutput_t = grad_net1
      else
        doutput_t = dzeros
      end
      table.insert(drnn_state[t], doutput_t)
      local dlst = clones_rnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(),
        unpack(rnn_state[t-1])}, drnn_state[t])
      drnn_state[t-1] = {}
      for k,v in pairs(dlst) do
        if k == 2 then -- k == 1 is gradient on x, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the
          -- derivatives of the state, starting at index 2. I know...
          drnn_pre:select(2, t):copy(v)
        elseif k>2 then
          drnn_state[t-1][k-2] = v
        end
      end
    end

    if opt.top_bidirection == 1 then
      local bidrnn_state = {[1] = clone_list(rnn_init_state, true)} -- true also zeros the clones
      -- perform back propagation through time (BPTT)
      for t = 1, x_length do
        local doutput_t
        if t == 1 then
          doutput_t = grad_net2
        else
          doutput_t = dzeros
        end

        table.insert(bidrnn_state[t], doutput_t)
        local dlst = clones_birnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(),
          unpack(birnn_state[t+1])}, bidrnn_state[t])
        bidrnn_state[t+1] = {}
        for k,v in pairs(dlst) do
          if k == 2 then -- k == 1 is gradient on x, which we dont need
            -- note we do k-1 because first item is dembeddings, and then follow the
            -- derivatives of the state, starting at index 2. I know...
            bidrnn_pre:select(2, t):copy(v)
          elseif k>2 then
            bidrnn_state[t+1][k-2] = v
          end
        end
      end
      drnn_pre:add(bidrnn_pre)
    end

    -- backward for mul_net
    grad_mul_net = pre_m:backward(attention_weight, drnn_pre)
  else
    local pre_m = TAGM_v2.pre_m
    local lstm = TAGM_v2.lstm2
    local rnn_init_state = TAGM_v2.rnn_init_state2
    local clones_rnn = TAGM_v2.clones_rnn2
    local clones_birnn = TAGM_v2.clones_birnn2
    local rnn_state = TAGM_v2.rnn_state2
    local birnn_state = TAGM_v2.birnn_state2
    local x_length = x:size(2)
    local pre_out = nil
    local drnn_pre = nil

    if opt.gpuid >= 0 then
     attention_weight = attention_weight:cuda()
     gradout = gradout:cuda()
    end
    local top_c2 = TAGM_v2.top_c2
    local hidden_z_value = TAGM_v2.hidden_z_value2
    local bidrnn_pre
    if opt.gpuid >= 0 then
      drnn_pre = torch.CudaTensor(opt.top_lstm_size, x_length):zero()
    else
       drnn_pre = torch.DoubleTensor(opt.top_lstm_size, x_length):zero()
    end
    if opt.top_bidirection == 1 and opt.gpuid>=0 then
      bidrnn_pre = torch.CudaTensor(opt.top_lstm_size, x_length):zero()
    elseif opt.top_bidirection == 1 and opt.gpuid==-1 then
      bidrnn_pre = torch.DoubleTensor(opt.top_lstm_size, x_length):zero()
    end
    local rnn_input = x
    if opt.gpuid >= 0 then
      pre_out = TAGM_v2.pre_out:cuda()
    else
      pre_out = TAGM_v2.pre_out
    end
    local top_c_grad = top_c2:backward(hidden_z_value, gradout)
    local grad_net1, grad_net2
    if opt.top_bidirection == 1 then
      grad_net1 = top_c_grad:sub(1, opt.top_lstm_size)
      grad_net2 = top_c_grad:sub(opt.top_lstm_size+1, -1)
    else
      grad_net1 = top_c_grad
    end
    local dzeros = torch.zeros(opt.top_lstm_size)
    if opt.gpuid >= 0 then
      dzeros = dzeros:cuda()
    end
    -- backward for rnn and birnn
    local drnn_state = {[x_length] = clone_list(rnn_init_state, true)} -- true also zeros the clones
    -- perform back propagation through time (BPTT)
    for t = x_length,1,-1 do
      local doutput_t
      if t == x_length then
        doutput_t = grad_net1
      else
        doutput_t = dzeros
      end
      table.insert(drnn_state[t], doutput_t)
      local dlst = clones_rnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(),
        unpack(rnn_state[t-1])}, drnn_state[t])
      drnn_state[t-1] = {}
      for k,v in pairs(dlst) do
        if k == 2 then -- k == 1 is gradient on x, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the
          -- derivatives of the state, starting at index 2. I know...
          drnn_pre:select(2, t):copy(v)
        elseif k>2 then
          drnn_state[t-1][k-2] = v
        end
      end
    end

    if opt.top_bidirection == 1 then
      local bidrnn_state = {[1] = clone_list(rnn_init_state, true)} -- true also zeros the clones
      -- perform back propagation through time (BPTT)
      for t = 1, x_length do
        local doutput_t
        if t == 1 then
          doutput_t = grad_net2
        else
          doutput_t = dzeros
        end

        table.insert(bidrnn_state[t], doutput_t)
        local dlst = clones_birnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(),
          unpack(birnn_state[t+1])}, bidrnn_state[t])
        bidrnn_state[t+1] = {}
        for k,v in pairs(dlst) do
          if k == 2 then -- k == 1 is gradient on x, which we dont need
            -- note we do k-1 because first item is dembeddings, and then follow the
            -- derivatives of the state, starting at index 2. I know...
            bidrnn_pre:select(2, t):copy(v)
          elseif k>2 then
            bidrnn_state[t+1][k-2] = v
          end
        end
      end
      drnn_pre:add(bidrnn_pre)
    end

    -- backward for mul_net
    grad_mul_net = pre_m:backward(attention_weight, drnn_pre)
  end

  return grad_mul_net

end

return TAGM_v2

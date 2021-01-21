
local model_utils = require 'util/model_utils'

local Multimodal_Fusion = {}


local function Attention_fusion(feature_dim)

  model = nn.Sequential()

  model:add(nn.Linear(feature_dim,feature_dim))
  model:add(nn.Tanh())

  model:add(nn.Linear(feature_dim, 1, false))
  model:add(nn.SoftMax())

  return model
end

function Multimodal_Fusion.model(loader, opt)

    att_fusion = Attention_fusion(loader.feature_dim)

    if opt.gpuid >= 0 then
      Multimodal_Fusion.att_fusion_model = att_fusion:cuda()
    else
      Multimodal_Fusion.att_fusion_model = att_fusion
    end
end

function Multimodal_Fusion.clone_model(loader, opt)
  print('cloning fusion model')
  local clones_att = model_utils.clone_many_times(Multimodal_Fusion.att_fusion_model, loader.max_time_series_length)

  Multimodal_Fusion.clones_att = clones_att
end


function Multimodal_Fusion.forward(x1, x2, opt)

  -- local hidden_init = Multimodal_Fusion.hidden_state
  local clones_att = Multimodal_Fusion.clones_att

  local x_fused_data
  if opt.gpuid >= 0 then
    x_fused_data = torch.zeros(x1:size(2), x1:size(1)):cuda()
  else
    x_fused_data = torch.zeros(x1:size(2), x1:size(1))
  end
  local x_fused
  local x_length = x1:size(2)
  for t=1,x_length do

    local x1_t = x1:narrow(2, t, 1)
    local x2_t = x2:narrow(2, t, 1)

    local att_1 = clones_att[t]:forward(x1_t:squeeze())
    local att_2 = clones_att[t]:forward(x2_t:squeeze())

    local x = torch.cat(x1_t,x2_t,2)
    local att = torch.cat(att_1, att_2, 2)

    att = torch.reshape(att, att:size(2), att:size(1))
    local mult
    if opt.gpuid >= 0 then
      mult = nn.MM():cuda()
    else
      mult = nn.MM()
    end
    x_fused = mult:forward({x, att})

    x_fused_data[t] = x_fused
  end
  x_fused_data = torch.reshape(x_fused_data, x_fused_data:size(2),x_fused_data:size(1))
  return x_fused_data
end

function Multimodal_Fusion.backward(opt, x1, x2, gradOut, loader)
  local clones_att = Multimodal_Fusion.clones_att
  local x_length = x1:size(2)
  -- perform back propagation through time (BPTT)
  for t=1,x_length do
    local x1_t = x1:narrow(2, t, 1)
    local x2_t = x2:narrow(2, t, 1)
    x1_t = torch.reshape(x1_t, x1_t:size(2), x1_t:size(1))
    x2_t = torch.reshape(x2_t, x2_t:size(2), x2_t:size(1))

    local grad_t = gradOut[t]
    local grads
    if opt.gpuid >= 0 then
      grads = torch.CudaTensor(1):fill(grad_t)
    else
      grads = torch.DoubleTensor(1):fill(grad_t)
    end

    local dlst1 = clones_att[t]:backward(x1_t:squeeze(), grads)
    local dlst2 = clones_att[t]:backward(x2_t:squeeze(), grads)

  end
  return

  -- body
end

return Multimodal_Fusion


local model_utils = require 'util/model_utils'

local Multimodal_Fusion = {}


local function Attention_fusion(feature_dim, n)

  model = nn.Sequential()

  model:add(nn.Linear(feature_dim,feature_dim))
  model:add(nn.Tanh())

  model:add(nn.Linear(feature_dim, 1, false))
  model:add(nn.SoftMax())

  return model
end

function Multimodal_Fusion.model(loader, opt)

    att_fusion_1 = Attention_fusion(loader.visual_feature_dim, 2)
    att_fusion_2 = Attention_fusion(loader.motion_feature_dim, 2)

    Multimodal_Fusion.att_fusion_model1 = att_fusion_1
  	Multimodal_Fusion.att_fusion_model2 = att_fusion_2
  	-- Multimodal_Fusion.hidden_state = hidden_state
end

function Multimodal_Fusion.clone_model(loader, opt)
  print('cloning fusion model')
  local clones_att1 = model_utils.clone_many_times(Multimodal_Fusion.att_fusion_model1, loader.max_time_series_length)
  local clones_att2 = model_utils.clone_many_times(Multimodal_Fusion.att_fusion_model2, loader.max_time_series_length)

  Multimodal_Fusion.clones_att1 = clones_att1
  Multimodal_Fusion.clones_att2 = clones_att2
end


function Multimodal_Fusion.forward(x1, x2, opt)
  local att_fusion1 = Multimodal_Fusion.att_fusion_model1
	local att_fusion2 = Multimodal_Fusion.att_fusion_model2


	-- local hidden_init = Multimodal_Fusion.hidden_state
  local clones_att1 = Multimodal_Fusion.clones_att1
	local clones_att2 = Multimodal_Fusion.clones_att2

  local x_fused_data = torch.zeros(x1:size(2), x1:size(1))

	local x_fused
	local x_length = x1:size(2)
	for t=1,x_length do

    local x1_t = x1:narrow(2, t, 1)
    local x2_t = x2:narrow(2, t, 1)

    local att_1 = clones_att1[t]:forward(x1_t:squeeze())
    local att_2 = clones_att2[t]:forward(x2_t:squeeze())

    -- local x2_transformed = torch.Tensor(4096, 1):zero()
    -- x2_transformed[{{1,100}}] = x2_t[{{1,100}}]
    local x2_transformed = torch.Tensor(x1_t:size(1), 1):zero()
    x2_transformed[{{1,x2_t:size(1)}}] = x2_t

    local x = torch.cat(x1:narrow(2, t, 1),x2_transformed,2)
    local att = torch.cat(att_1, att_2, 2)

    -- att = nn.View(2, 1):forward(att)
    att = torch.reshape(att, att:size(2), att:size(1))
    local mult = nn.MM()
    x_fused = mult:forward({x, att})

    x_fused_data[t] = x_fused
  end
  x_fused_data = torch.reshape(x_fused_data, x_fused_data:size(2),x_fused_data:size(1))
  return x_fused_data
end

function Multimodal_Fusion.backward(opt, x1, x2, gradOut, loader)
  local att_fusion1 = Multimodal_Fusion.att_fusion_model1
  local att_fusion2 = Multimodal_Fusion.att_fusion_model2
  local clones_att1 = Multimodal_Fusion.clones_att1
  local clones_att2 = Multimodal_Fusion.clones_att2

  local x_length = x1:size(2)
  -- perform back propagation through time (BPTT)
  for t=1,x_length do
    -- local x = torch.cat(x1:narrow(2, t, 1),x2:narrow(2, t, 1),2)
    -- x = torch.reshape(x, x:size(2), x:size(1))

    -- grad_t = gradOut[t]
    -- local grads = torch.Tensor(2, 1):fill(grad_t)

    -- local dlst = clones_att[t]:backward(x, grads)
    local x1_t = x1:narrow(2, t, 1)
    local x2_t = x2:narrow(2, t, 1)
    x1_t = torch.reshape(x1_t, x1_t:size(2), x1_t:size(1))
    x2_t = torch.reshape(x2_t, x2_t:size(2), x2_t:size(1))

    local grad_t = gradOut[t]
    local grads = torch.Tensor(1):fill(grad_t)

    local dlst1 = clones_att1[t]:backward(x1_t:squeeze(), grads)
    local dlst2 = clones_att2[t]:backward(x2_t:squeeze(), grads)

  end

  -- body
end

return Multimodal_Fusion

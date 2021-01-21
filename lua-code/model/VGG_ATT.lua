
--[[
Taken the hidden-unit values of the hidden units from RNN modules, This Top_Net performs the following operations:
]]--

require 'nn'
require 'rnn'
require 'loadcaffe'

-- local CNN = require 'model.my_CNN'
-- local model_utils = require 'util/model_utils'

local function make_layers(modelType)
    local cfg = {}
    cfg = {64, 64, 128, 128, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 'M', 512, 'M'}

    local l1 = nn.Sequential()
    local l2 = nn.Sequential()
    local l3 = nn.Sequential()
    local conv_out = nn.Sequential()
    do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
        if k <= 8 then
         if v == 'M' then
            l1:add(nn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            -- local batch2d = nn.SpatialBatchNormalization(oChannels,1e-3)
            l1:add(conv3)
            l1:add(nn.ReLU(true))
            iChannels = oChannels;
         end
       elseif k <= 12 then
        if v == 'M' then
            l2:add(nn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            -- local batch2d = nn.SpatialBatchNormalization(oChannels,1e-3)
            l2:add(conv3)
            l2:add(nn.ReLU(true))
            iChannels = oChannels;
          end
       elseif k <= 16 then
        if v == 'M' then
            l3:add(nn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            -- local batch2d = nn.SpatialBatchNormalization(oChannels,1e-3)
            l3:add(conv3)
            l3:add(nn.ReLU(true))
            iChannels = oChannels;
          end
       else
        if v == 'M' then
            conv_out:add(nn.SpatialMaxPooling(2,2,5,5))
         else
            local oChannels = v;
            local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            -- local batch2d = nn.SpatialBatchNormalization(oChannels,1e-3)
            conv_out:add(conv3)
            conv_out:add(nn.ReLU(true))
            iChannels = oChannels;
          end
       end
      end
    end
    return l1,l2,l3,conv_out
end

local function _weighted_combine(l, att_map)
    local l = l:view(l:size(2),l:size(3),l:size(4))
    local att_map = att_map:view(l:size(2),l:size(3))

    for i=1,l:size(1) do
      l[i] = l[i] * att_map
    end

    l = l:view(l:size(1), l:size(2)*l:size(3))
    weights = torch.sum(l,2)

    return weights
  -- body
end

local function _compatibility_fn( l, g, level )
  ---  self.mode == 'dp':
  local g_res = g:view(g:size(1))
  local l_res = l:view(l:size(2),l:size(3),l:size(4))

  for i=1,g_res:size(1) do
    l_res[i] = torch.mul(l_res[i], g_res[i])
  end

  att = torch.sum(l_res,1)
  size = att:size()

  att = att:view(att:size(1), att:size(2)*att:size(3))

  probs = nn.SoftMax()
  probs:cuda()
  att = probs:forward(att)
  att = att:view(size)

  return att
end


local VGG_ATT = {}

function VGG_ATT.net(loader, opt)

  local class_n = loader.class_size
  local l1, l2, l3, conv_out
  l1,l2,l3,conv_out = make_layers('vgg_att')
  l1:cuda()
  l2:cuda()
  l3:cuda()
  conv_out:cuda()

  VGG_ATT.l1 = l1
  VGG_ATT.l2 = l2
  VGG_ATT.l3 = l3
  VGG_ATT.conv_out=conv_out
  VGG_ATT.fc1 = nn.Linear(512, 512)
  VGG_ATT.fc1:cuda()

  VGG_ATT.fc1_l1 = nn.Linear(512, 256)
  VGG_ATT.fc1_l1:cuda()
  VGG_ATT.fc1_l2 = nn.Linear(512, 512)
  VGG_ATT.fc1_l2:cuda()
  VGG_ATT.fc1_l3 = nn.Linear(512, 512)
  VGG_ATT.fc1_l3:cuda()

  -- VGG_ATT.fc2 = nn.Linear(256 + 512 + 512, class_n)
  -- VGG_ATT.fc2:cuda()

  local params_l1 = VGG_ATT.l1:getParameters():nElement()
  local params_l2 = VGG_ATT.l2:getParameters():nElement()
  local params_l3 = VGG_ATT.l3:getParameters():nElement()
  local params_fc1 = VGG_ATT.fc1:getParameters():nElement()
  local params_fc1_l1 = VGG_ATT.fc1_l1:getParameters():nElement()
  local params_fc1_l2 = VGG_ATT.fc1_l2:getParameters():nElement()
  local params_fc1_l3 = VGG_ATT.fc1_l3:getParameters():nElement()
  -- local params_fc2 = VGG_ATT.fc2:getParameters():nElement()
  -- VGG_ATT.params_size = params_l1+params_l2+params_l3+params_fc1+params_fc1_l1+params_fc1_l2+params_fc1_l3+params_fc2
  VGG_ATT.params_size = params_l1+params_l2+params_l3+params_fc1+params_fc1_l1+params_fc1_l2+params_fc1_l3
  print('number of parameters in the top cnn model: ' .. VGG_ATT.params_size)
end


function VGG_ATT.forward(x, opt, flag)
  local l1 = VGG_ATT.l1
  local l2 = VGG_ATT.l2
  local l3 = VGG_ATT.l3
  local conv_out = VGG_ATT.conv_out
  local fc1 = VGG_ATT.fc1
  -- local fc2 = VGG_ATT.fc2
  local fc1_l1 = VGG_ATT.fc1_l1
  local fc1_l2 = VGG_ATT.fc1_l2
  local fc1_l3 = VGG_ATT.fc1_l3
  local g = nil
  local l1_ = {}
  local l2_ = {}
  local l3_ = {}
  local fc1_ = {}
  local fc1_l1_ = nil
  local fc1_l2_ = nil
  local fc1_l3_ = nil
  local conv_out_ = {}
  local x_length = nil
  local output_v
  local outputs = torch.CudaTensor(30, 256 + 512 + 512)

  x = x:resize(x:size(4),x:size(1),x:size(2),x:size(3))

  for i=1,30 do
    local x_resized = x[i]:resize(1,x[i]:size(1),x[i]:size(2),x[i]:size(3))
    x_resized = x_resized:cuda()
    l1_[i] = l1:forward(x_resized)
    l2_[i] = l2:forward(l1_[i])
    l3_[i] = l3:forward(l2_[i])

    conv_out_[i] = conv_out:forward(l3_[i])

    fc1_[i] = fc1:forward(conv_out_[i]:resize(conv_out_[i]:size(2)))
    fc1_l1_ = fc1_l1:forward(fc1_[i])
    fc1_l2_= fc1_l2:forward(fc1_[i])
    fc1_l3_ = fc1_l3:forward(fc1_[i])

    att1 = _compatibility_fn(l1_[i], fc1_l1_, 1)
    att2 = _compatibility_fn(l2_[i], fc1_l2_, 2)
    att3 = _compatibility_fn(l3_[i], fc1_l3_, 3)

    g1 = _weighted_combine(l1_[i], att1)
    g2 = _weighted_combine(l2_[i], att2)
    g3 = _weighted_combine(l3_[i], att3)

    g = torch.cat(torch.cat(g1, g2, 1), g3, 1)

    g = g:view(g:size(1))
    -- out = fc2:forward(g)
    --
    -- outputs = out
    outputs[i] = g
  end
  if flag == 'training' then
    return outputs,l1_, l2_, l3_, conv_out_, fc1_, g
  else
    return outputs
  end

end

function VGG_ATT.backward(x, l1_, l2_, l3_, conv_out_, fc1_, g, opt, gradout)
  local l1 = VGG_ATT.l1
  local l2 = VGG_ATT.l2
  local l3 = VGG_ATT.l3
  local fc1 = VGG_ATT.fc1
  -- local fc2 = VGG_ATT.fc2
  local conv_out = VGG_ATT.conv_out
  local fc1_l1 = VGG_ATT.fc1_l1
  local fc1_l2 = VGG_ATT.fc1_l2
  local fc1_l3 = VGG_ATT.fc1_l3
  local x_length = nil
  local output_v
  local outputs

  for i=30,1,-1 do

    local x_resized = x[i]:resize(1,3,x[i]:size(2),x[i]:size(3))
    x_resized = x_resized:cuda()

    -- grads_fc2 = fc2:backward(g,gradout)
    local grads = torch.CudaTensor(512):fill(gradout[i])

    fc1_l1_grads = fc1_l1:backward(fc1_[i],grads[{{1,256}}])
    fc1_l2_grads = fc1_l2:backward(fc1_[i],grads)
    fc1_l3_grads = fc1_l3:backward(fc1_[i],grads)

    fc1_grads = fc1:backward(conv_out_[i], fc1_l3_grads)

    conv_out_grads = conv_out:backward(l3_[i], fc1_grads:view(1,fc1_grads:size(1),1,1))

    l3_grads = l3:backward(l2_[i],conv_out_grads)
    l2_grads = l2:backward(l1_[i],l3_grads)

    l1_grads = l1:backward(x_resized,l2_grads)
  end

  return grad_net


end


return VGG_ATT

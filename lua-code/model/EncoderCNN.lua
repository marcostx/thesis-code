
--[[
Taken the hidden-unit values of the hidden units from RNN modules, This Top_Net performs the following operations:
]]--

require 'nn'
require 'loadcaffe'

if not paths.dirp('model_weights') then
  print('=> Downloading VGG 19 model weights')
  os.execute('mkdir model_weights')
  local caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'
  local proto_url = 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt'
  os.execute('wget --output-document model_weights/VGG_ILSVRC_16_layers.caffemodel ' .. caffemodel_url)
  os.execute('wget --output-document model_weights/VGG_ILSVRC_16_layers_deploy.prototxt ' .. proto_url)
end


local CNN = require 'model.my_CNN'
local model_utils = require 'util/model_utils'

local EncoderCNN = {}

function EncoderCNN.net(loader, opt)

  local class_n = loader.class_size
  -- local cnn

  -- cnn = CNN.cnn(class_n)
  -- for indx,module in pairs(cnn:findModules('nn.SpatialConvolution')) do
  --   module.weight:normal(0,math.sqrt(1/(module.kW*module.kH*module.nOutputPlane)))
  -- end
  local proto = 'model_weights/VGG_ILSVRC_16_layers_deploy.prototxt'
  local caffemodel = 'model_weights/VGG_ILSVRC_16_layers.caffemodel'

  local cnn = loadcaffe.load(proto, caffemodel)
  for i=1,5 do
    cnn.modules[#cnn.modules] = nil
  end
  EncoderCNN.cnn = cnn:cuda()

  EncoderCNN.params_size = cnn:getParameters():nElement()
  print('number of parameters in the top cnn model: ' .. EncoderCNN.params_size)
end

function EncoderCNN.init_params(opt)

end

function EncoderCNN.forward(x, opt, flag)
  local net = EncoderCNN.cnn
  local x_length = nil
  local output_v
  local cnn_embeddings

  x = x:resize(x:size(4), x:size(1),x:size(2),x:size(3))
  x_length = x:size(1)
  cnn_embeddings = torch.CudaTensor(x_length,4096)

  for i=1,x_length do

    local x_resized = x[i]:resize(1,3,x[i]:size(2),x[i]:size(3))

    x_resized = x_resized:cuda()
    output_v = net:forward(x_resized)
    cnn_embeddings[i] = output_v
  end
  return cnn_embeddings
end

function EncoderCNN.backward(x, opt, gradout)
  local net = EncoderCNN.cnn
  local x_length = nil
  local grad_net

  --x = x:resize(x:size(4), x:size(1),x:size(2),x:size(3))
  x_length = x:size(1)

  for i=1,x_length do
    local x_resized = x[i]:resize(1,3,x[i]:size(2),x[i]:size(3))
    local grads = torch.CudaTensor(4096):fill(gradout[i])
    grad_net = net:backward(x_resized:cuda(), grads)
  end

  return grad_net

end


return EncoderCNN

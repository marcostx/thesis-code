
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

local Top_CNN = {}

function Top_CNN.net(loader, opt)

  local class_n = loader.class_size
  -- local cnn

  -- cnn = CNN.cnn(class_n)
  -- for indx,module in pairs(cnn:findModules('nn.SpatialConvolution')) do
  --   module.weight:normal(0,math.sqrt(1/(module.kW*module.kH*module.nOutputPlane)))
  -- end
  local proto = 'model_weights/VGG_ILSVRC_16_layers_deploy.prototxt'
  local caffemodel = 'model_weights/VGG_ILSVRC_16_layers.caffemodel'

  local cnn = loadcaffe.load(proto, caffemodel)
  for i=1,7 do
    cnn.modules[#cnn.modules] = nil 
  end
  print(cnn)
  os.exit(1)
  Top_CNN.cnn = cnn:cuda()

  Top_CNN.params_size = cnn:getParameters():nElement() 
  print('number of parameters in the top cnn model: ' .. Top_CNN.params_size)
end

function Top_CNN.init_params(opt)

end

function Top_CNN.clone_model(loader) 
-- pa  
end

function Top_CNN.forward(x, opt, flag)
  local net = Top_CNN.cnn
  local x_length = nil
  local output_v
	
  x = x:resize(x:size(4), x:size(1),x:size(2),x:size(3))
  x_length = x:size(1)  
  
  for i=10,x_length do
  
    local x_resized = x[i]:resize(1,3,x[i]:size(2),x[i]:size(3))
     
    x_resized = x_resized:cuda() 
    output_v = net:forward(x_resized)
  end
  
  return output_v
end

function Top_CNN.backward(x, opt, gradout, loader)
  local net = Top_CNN.cnn
  local x_length = nil
  local grad_net

  --x = x:resize(x:size(4), x:size(1),x:size(2),x:size(3))
  x_length = x:size(1)
  
  for i=10,x_length do
    local x_resized = x[i]:resize(1,3,x[i]:size(2),x[i]:size(3))
    grad_net = net:backward(x_resized:cuda(), gradout)
  end
 
  return grad_net

end


return Top_CNN




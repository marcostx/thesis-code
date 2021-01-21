
--[[
Taken the hidden-unit values of the hidden units from RNN modules, This Top_Net performs the following operations:
]]--

require 'nn'
require 'rnn'
require 'loadcaffe'


if not paths.dirp('model_weights') then
  print('=> Downloading VGG 19 model weights')
  os.execute('mkdir model_weights')
  local caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'
  local proto_url = 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt'
  os.execute('wget --output-document model_weights/VGG_ILSVRC_16_layers.caffemodel ' .. caffemodel_url)
  os.execute('wget --output-document model_weights/VGG_ILSVRC_16_layers_deploy.prototxt ' .. proto_url)
end


local function cnn(nb_outputs)
  
  local proto = 'model_weights/VGG_ILSVRC_16_layers_deploy.prototxt'
  local caffemodel = 'model_weights/VGG_ILSVRC_16_layers.caffemodel'

  local cnn = loadcaffe.load(proto, caffemodel)
  for i=1,2 do
    cnn.modules[#cnn.modules] = nil 
  end
  cnn:add(nn.RecLSTM(4096, 32))
  cnn:add(nn.Linear(32, 2))
  cnn:add(nn.LogSoftMax())
  cnn:cuda()

  return cnn
end


local ConvLSTM = {}

function ConvLSTM.net(loader, opt)

  local class_n = loader.class_size
  local cnvlstm

  cnvlstm = cnn(opt.class_size)
  cnvlstm:cuda()

  ConvLSTM.convlstm = cnvlstm

  ConvLSTM.params_size = cnvlstm:getParameters():nElement()
  print('number of parameters in the top cnn model: ' .. ConvLSTM.params_size)
end

function ConvLSTM.init_params(opt)

end

function ConvLSTM.forward(x, opt, flag)
  local net = ConvLSTM.convlstm
  local x_length = nil
  local output_v
  local outputs
  x = x:resize(x:size(4), x:size(1),x:size(2),x:size(3))

  for i=1,30 do

    local x_resized = x[i]:resize(1,3,x[i]:size(2),x[i]:size(3))

    x_resized = x_resized:cuda()
    output_v = net:forward(x_resized)
    outputs = output_v
    collectgarbage()
  end
  


  return outputs
end

function ConvLSTM.backward(x, opt, gradout)
    local net = ConvLSTM.convlstm
  local x_length = nil
  local grad_net

  --x = x:resize(x:size(4), x:size(1),x:size(2),x:size(3))
  
  for i=1,30 do
    local x_resized = x[i]:resize(1,3,x[i]:size(2),x[i]:size(3))
    grad_net = net:backward(x_resized:cuda(), gradout)
    collectgarbage()
  end
  
 
  return grad_net
  

end


return ConvLSTM

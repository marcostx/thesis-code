--require 'image'
--require 'ffmpeg'
npy4th = require 'npy4th'

function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

-----------------------------------------------------
------------ convert npy file to t7 -----------------
-----------------------------------------------------
function npy_to_t7()
  print("Converting dataset (Numpy -> T7)")
  
  local trainx_filename = '/work/src/train-flow-media_x.npy'
  local trainy_filename = '/work/src/train-flow-media_y.npy'
  local validx_filename = '/work/src/val-flow-media_x.npy'
  local validy_filename = '/work/src/val-flow-media_y.npy'
  local output_file = '/work/src/rwf-efficientnet-fine.t7'

  trainx = npy4th.loadnpy(trainx_filename)
  validx = npy4th.loadnpy(validx_filename)
  trainy = npy4th.loadnpy(trainy_filename)
  validy = npy4th.loadnpy(validy_filename)

  trainx = trainx:resize(trainx:size(1),1,trainx:size(3),trainx:size(2))
  validx = validx:resize(validx:size(1),1,validx:size(3),validx:size(2))
  trainy = trainy:resize(trainy:size(1))
  validy = validy:resize(validy:size(1))

  dataset = {
        train_X = trainx,
        train_y = trainy,
        valid_X = validx,
        valid_y = validy
      }

  print("saving dataset")
  torch.save(output_file,dataset)
end
-----------------------------------------------------
-- convert cross validation dataset npy file to t7 --
-----------------------------------------------------
function cv_npy_to_t7()
  print("Converting dataset (Numpy -> T7)")
  local output_file = '/work/src/hockey-efficientnet.t7'
  splits_arr={}

  local nSplits = 5
  for splt=1,nSplits do
    local trainx_filename = string.format('/work/src/%d/train-hockey_x.npy',splt)
    local trainy_filename = string.format('/work/src/%d/train-hockey_y.npy',splt)
    local validx_filename = string.format('/work/src/%d/test-hockey_x.npy',splt)
    local validy_filename = string.format('/work/src/%d/test-hockey_y.npy',splt)

    trainx = npy4th.loadnpy(trainx_filename)
    validx = npy4th.loadnpy(validx_filename)
    trainy = npy4th.loadnpy(trainy_filename)
    validy = npy4th.loadnpy(validy_filename)

    trainx = trainx:resize(trainx:size(1),1,trainx:size(3),trainx:size(2))
    validx = validx:resize(validx:size(1),1,validx:size(3),validx:size(2))
    trainy = trainy:resize(trainy:size(1))
    validy = validy:resize(validy:size(1))
    print(trainx:size())
    print(validx:size())

    dataset = {
          train_X = trainx,
          train_y = trainy,
          valid_X = validx,
          valid_y = validy
        }
    splits_arr[splt] = dataset
  end

  print("saving dataset")
  torch.save(output_file,splits_arr)
end

function unimodal_to_multimodal()
  print("merge rg and flow datasets into one t7 file")
  -- mediaeval 2015
  -- 
  local data_raw = '/work/src/data/rwf-train-val-flow-efficientnet.t7'
  local data_of = '/work/src/data/rwf-train-val-efficientnet.t7'
  local output_file = '/work/src/rwf-train-val-efficientnet-multimodal.t7'
  local data_rgb = torch.load(data_raw)
  local data_flow = torch.load(data_of)
  print(data_rgb)
  print(data_flow)

  dataset = {
    train_X_visual = data_rgb.train_X,
    train_X_motion = data_flow.train_X,
    train_y_visual = data_rgb.train_y,
    train_y_motion = data_flow.train_y,
    test_X_visual = data_rgb.valid_X,
    test_X_motion = data_flow.valid_X,
    test_y_visual = data_rgb.valid_y,
    test_y_motion = data_flow.valid_y
  }

  -- hockey dataset
  -- local data_raw = '/work/src/data/hockey-efficientnet.t7'
  -- local data_of = '/work/src/data/hockey-flow-efficientnet.t7'
  -- local output_file = '/work/src/hockey-efficientnet-multimodal.t7'
  -- local data_rgb = torch.load(data_raw)
  -- local data_flow = torch.load(data_of)
  -- splits_arr={}
  -- print(data_rgb)
  -- print(data_flow)

  -- local nSplits = 5
  -- for splt=1,nSplits do
  --   dataset = {
  --     train_X_visual = data_rgb[splt].train_X,
  --     train_X_motion = data_flow[splt].train_X,
  --     train_y_visual = data_rgb[splt].train_y,
  --     train_y_motion = data_flow[splt].train_y,
  --     test_X_visual = data_rgb[splt].valid_X,
  --     test_X_motion = data_flow[splt].valid_X,
  --     test_y_visual = data_rgb[splt].valid_y,
  --     test_y_motion = data_flow[splt].valid_y
  --   }
  --   splits_arr[splt] = dataset
  -- end

  print("saving dataset")
  torch.save(output_file,dataset)
end
print(npy_to_t7())

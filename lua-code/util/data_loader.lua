local path = require 'pl.path'
local table_operation = require 'util.table_operation'

local data_loader = {}
data_loader.__index = data_loader


function data_loader.create(opt)
  local self = {}
  setmetatable(self, data_loader)

  self:load_unimodal_data(opt)

  self.batch_ix = 1
  --self.rand_order = torch.randperm(self.nTrain)
  self.rand_order = nil
  --self.rand_order=nil
  print('data load done. ')
  print('before collectgarbage: ', collectgarbage("count"))
  collectgarbage()
  print('after collectgarbage: ', collectgarbage("count"))
  --  os.exit()
  return self
end

-- Return the tensor of index of the data with batch size
function data_loader:get_next_train_batch(batch_size)
  self.previous_batch_ix = self.batch_ix
  local set_size = self.nTrain;
  local rtn_index = torch.zeros(batch_size)
  for i = 1, batch_size do
    local temp_ind = i + self.batch_ix - 1
    if temp_ind > set_size then -- cycle around to beginning
      temp_ind = temp_ind - set_size
    end
    rtn_index[i] = self.rand_order[temp_ind]
  end
  self.batch_ix = self.batch_ix + batch_size;
  -- cycle around to beginning
  if self.batch_ix >set_size then
    self.batch_ix = 1
    -- randomize the order of the training set
    self.rand_order = torch.randperm(self.nTrain)
  end
  return rtn_index;
end

function data_loader:load_data_by_index_t7(opt)
  print('loading ', opt.data_set, ' data: ')
  local data_dir = nil
  local data = nil

  data_dir = '../data/violentFlows.t7'
  data = torch.load(data_dir)

  self.train_X = data.train_X
  self.validation_X = data.validation_X
  self.test_X = data.test_X
  self.train_T = data.train_T
  self.validation_T = data.validation_T
  self.test_T = data.test_T
  self.nTrain = self.train_T:size(1)
  self.nValidation = self.validation_T:size(1)
  self.nTest = self.test_T:size(1)
  local data_size = self.train_T:nElement()+self.validation_T:nElement()+self.test_T:nElement()
  self.class_size = data.train_T:max()+1
  print('class number: ', self.class_size)
  print('training size: ', self.train_T:size(1))
  print('validation size: ', self.validation_T:size(1))
  print('test size: ', self.test_T:size(1))

  self.max_time_series_length = self.train_X:size(4)
  print('The max length of the time series in this data set: ', self.max_time_series_length)

  self.feature_dim = self.train_X:size(3)
  print('feature dimension: ', self.feature_dim)

end

function data_loader:load_unimodal_data(opt)

  print('loading ', opt.data_set, ' data: ')
  local data_dir = nil
  self.data = {}

  data_dir = opt.data_set
  self.data = torch.load(data_dir)

  self.train_X_visual = nil
  self.train_X_motion = nil
  self.test_X_visual = nil
  self.test_X_motion = nil
  self.train_X = nil
  self.valid_X = nil
  self.train_y = nil
  self.valid_y = nil
  self.nTrain = nil
  self.nTest = nil
  self.class_size = 2
  self.nSplits = 5

  print('class number: ', self.class_size)
  -- print('training size: ', self.train_T:size(1))
  -- print('validation size: ', self.validation_T:size(1))
  -- print('test size: ', self.test_T:size(1))

  -- self.max_time_series_length = self.data.train_X:size(4)
  self.max_time_series_length = 150
  self.visual_feature_dim = 1536
  self.motion_feature_dim = 1536
  print('The max length of the time series in this data set: ', self.max_time_series_length)

  if opt.use_visual then
    -- self.feature_dim = self.data.train_X:size(3)
    self.feature_dim = 1536
   end

  print('visual feature dimension: ', self.feature_dim)
  --print('fusion feature dimension', self.feature_dim)
end

-- function data_loader:load_data_mediaeval(opt)

--   print('loading ', opt.data_set, ' data: ')
--   local data_dir = nil
--   self.data = {}

--   data_dir = opt.data_set
--   self.data = torch.load(data_dir)


--   self.train_X_visual = self.data.train_X_visual
--   self.test_X_visual = self.data.test_X_visual
--   self.train_T = self.data.train_T
--   self.test_T = self.data.test_T
--   self.nTrain = self.train_T:size(1)
--   self.nTest = self.test_T:size(1)
--   self.class_size = self.data.train_T:max()+1
--   self.nSplits = 1
--   print('class number: ', self.class_size)
--   print('training size: ', self.train_T:size(1))
--   print('test size: ', self.test_T:size(1))

--   self.max_time_series_length = self.data.train_X_visual:size(4)
--   print('The max length of the time series in this data set: ', self.max_time_series_length)

--   self.feature_dim = self.data.train_X_visual:size(3)
--   print('feature dimension: ', self.feature_dim)
-- end

function data_loader:load_multimodal_data_cross_val(opt)
  print('loading ', opt.data_set, ' data: ')
  local data_dir = nil
  self.data = {}

  data_dir = opt.data_set
  self.data = torch.load(data_dir)

  self.train_X_visual = nil
  self.train_X_motion = nil
  self.test_X_visual = nil
  self.test_X_motion = nil
  self.train_T = nil
  self.test_T = nil
  self.nTrain = nil
  self.nTest = nil
  self.class_size = self.data[1].train_T:max()+1
  self.nSplits = 5
  

  print('class number: ', self.class_size)
  -- print('training size: ', self.train_T:size(1))
  -- print('validation size: ', self.validation_T:size(1))
  -- print('test size: ', self.test_T:size(1))

  self.max_time_series_length = self.data[1].train_X_visual:size(4)
  print('The max length of the time series in this data set: ', self.max_time_series_length)

  self.visual_feature_dim = self.data[1].train_X_visual:size(3)
  self.motion_feature_dim = self.data[1].train_X_motion:size(3)
  if opt.early then
    self.feature_dim = self.visual_feature_dim + self.motion_feature_dim
    print(" Early fusion dimension:", self.feature_dim)
  else
    self.feature_dim = math.max(self.visual_feature_dim, self.motion_feature_dim)
  end
  print('visual feature dimension: ', self.visual_feature_dim)
  print('motion feature dimension: ', self.motion_feature_dim)
  print('fusion feature dimension', self.feature_dim)
end

return data_loader

local stringx = require('pl.stringx')
local file = require('pl.file')

local path = "./Data/"

--Function to load training data from the text file
local function load_data1(fname)
   local k1 = 1
   local len1 = 0
   local data = file.read(fname)
   data = stringx.replace(data,'\n',',')
   data = stringx.split(data,',')
   print(string.format("Loading %s, size of data = %d", fname, #data))
   local x = torch.LongTensor((#data/3),3)
   for i = 1, #data do
      data[i]= tonumber(data[i])
   end
      len1 = #data 
    for i = 1, (len1/3) do
      for j=1,3 do  
          x[i][j]= data[k1]
          k1=k1+1
      end
   end
   return x
end

--Function to load testing data from the text file
local function load_data2(fname)
   local k2 = 1
   local len2 = 0
   local data = file.read(fname)
   data = stringx.replace(data,'\n',',')
   data = stringx.split(data,',')
   print(string.format("Loading %s, size of data = %d", fname, #data))
   local x = torch.Tensor((#data/3),3)
   for i = 1, #data do
      data[i]= tonumber(data[i])
   end
      len2 = #data 
    for i = 1, (len2/3) do
      for j=1,3 do  
          x[i][j]= data[k2]
          k2=k2+1
      end
   end
   return x
end

--Function to acquire training data
local function traindataset()
   local x = load_data1(path .. "train.dta")
   return x
end

--Function to acquire training labels
local function trainlabels()
  local y = torch.zeros((9900*133))
  for a = 1, (9900*133) do
  y[a]=math.ceil(a/9900) 
  end
  return y 
end

--Function to acquire testing data
local function testdataset()
   local x = load_data2(path .. "test.dta")
   return x
end

--Function to acquire testing labels
local function testlabels()
  local y = torch.zeros((2800*133))
  for c = 1, (2800*133) do
  y[c]=math.ceil(c/2800) 
  end 
  return y
end

return {traindataset = traindataset,
        trainlabels = trainlabels,
        testdataset=testdataset,
        testlabels=testlabels}
require('dpnn')
require('rnn')
require('nngraph')
require('optim')
data = require('data')
nninit = require('nninit')

gpu=1
if gpu>0 then
  print("CUDA ON")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(gpu)
end

--Parameters
nPredict =1
batchSize = 16
rho = 100
hiddenSize = 800
inputSize = 3
outputSize = 133
lr = 0.001

--Loading the testing data
test_data = data.testdataset()
test_labels = data.testlabels()

if gpu>0 then
  test_data = test_data:cuda()
  test_labels = test_labels:cuda()
end

--Loading the trained model
paths = require 'paths'
filename  = paths.concat(paths.cwd(),'Train6.net')
model = torch.load(filename)

--Evaluating the trained model
model:evaluate()
local correct=0
local iters = 1
while iters<=nPredict do
  for i=1 ,133 do
    local outputs = torch.zeros(1,outputSize)
    if gpu>0 then
      outputs=outputs:cuda()
    end  
    for j=1,100 do
      local start = {}
      local v = math.ceil(math.random()*2700)
      local s = (i-1)*2800 + v
      table.insert(start,s)
      start = torch.Tensor(start)
      if gpu>0 then
        start=start:cuda()
      end

      local inputs_test = {}
      for step=1,rho do
        inputs_test[step] = test_data:index(1, start):view(1,inputSize)
        start:add(1)
      end

      model:zeroGradParameters()
      local outputs_test = model:forward(inputs_test)
      outputs = outputs + outputs_test[rho]

    end  
    local _,position = torch.max(outputs,2)
    print(i)
    print(position)
    a=i
    b=position:sum()
    if (a == b) then
      correct = correct +1
      print(correct)
    end
  end
  iters = iters+1
end
accuracy = ((correct*100)/(nPredict*133))


print (accuracy)
print(correct)

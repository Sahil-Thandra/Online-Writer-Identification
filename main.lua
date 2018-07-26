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
nIters = 3990
nPredict = 30
batchSize = 16   
rho = 100
hiddenSize = 800
inputSize = 3
outputSize = 133
lr = 0.001

--BiRNN Architecture
lstm = nn.FastLSTM(hiddenSize, hiddenSize,rho)
brnn = nn.Sequential()
   :add(nn.Linear(inputSize, hiddenSize))
   :add(lstm)
   :add(nn.NormStabilizer())
brnn = nn.BiSequencer(brnn)

rnn = nn.Sequential()
   :add(brnn) 
   :add(nn.Sequencer(nn.Linear(hiddenSize*2, outputSize))) 
print(rnn)


if gpu>0 then
  rnn=rnn:cuda()
end

--Loss Function
criterion = nn.CrossEntropyCriterion()

if gpu>0 then
  criterion=criterion:cuda()
end

--Initialising the weights and biases
rnn:getParameters():normal(0,0.01)
lstm.i2g:init({'bias', {{2*hiddenSize+1, 3*hiddenSize}}}, nninit.constant, 5)
paramx,paramdx = rnn:getParameters()

--Loading the training and testing data
train_data = data.traindataset()
train_labels = data.trainlabels()
test_data = data.testdataset()
test_labels = data.testlabels()

if gpu>0 then
  train_data = train_data:cuda()
  train_labels = train_labels:cuda()
  test_data = test_data:cuda()
  test_labels = test_labels:cuda()
end

--Initialising the gradients to zeroes
gradOutputsZeroed = {}
for step=1,rho do
  gradOutputsZeroed[step] = torch.zeros(batchSize,outputSize)
  if gpu>0 then
    gradOutputsZeroed[step] = gradOutputsZeroed[step]:cuda()
  end
end

--Function to convert tensor to a table
function tensor2Table(inputTensor)
   local outputTable = {}
   for t = 1, inputTensor:size(1) do outputTable[t] = inputTensor[t] end
   return outputTable
end

--Function to convert table to a tensor
function table2Tensor(tx)
  local tensorSize = tx[1]:size()
  local tensorSizeTable = {-1}
  for i=1,tensorSize:size(1) do
    tensorSizeTable[i+1] = tensorSize[i]
  end
  merge=nn.Sequential()
    :add(nn.JoinTable(1))
    :add(nn.View(unpack(tensorSizeTable)))

  return merge:forward(tx)
end


--Function to evaluate loss and gradients
function feval()
    
    rnn:training()
    rnn:zeroGradParameters()
    outputs_table = rnn:forward(inputs)
    outputs = table2Tensor(outputs_table)
    targets_tensor = table2Tensor(targets)
    if gpu>0 then
      outputs = outputs:cuda()
      targets_tensor = targets_tensor:cuda()
    end
    err = criterion:forward(outputs[rho], targets_tensor[rho])
    gradOutputs = criterion:backward(outputs[rho], targets_tensor[rho])
    gradOutputsZeroed[rho] = gradOutputs
    rnn:backward(inputs, gradOutputsZeroed)
    
  return err, paramdx
end 


--Training the BiRNN model
trainError = 0
iteration = 1
while iteration <= nIters do
   
  offsets = {}

  for i=1,batchSize do
      o = math.ceil(math.random()* 9800 )
      
      if o>5500 then
        if ((o-5400)%5500)<=rho then
          o = o + rho +1
        end    
      end
      
      s = ((((iteration-1)+(8*(i-1)))%133)*9900)+o

      table.insert(offsets, s)
  end


  offsets = torch.Tensor(offsets)
  if gpu>0 then
      offsets=offsets:cuda()
  end

   inputs={}
   targets={}
   for step=1,rho do
      inputs[step] = train_data:index(1, offsets):view(batchSize,inputSize)
      targets[step] = train_labels:index(1, offsets):view(batchSize,1)
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > train_data:size(1) then
            offsets[j] = 1
         end
      end
   end
      
  paramx, Error = optim.adam(feval,paramx)
  trainError = trainError + Error[1]

  print(string.format("Iteration %d ; AdamOpt err = %f ", iteration, Error[1]))

  iteration = iteration + 1
end
trainError = trainError/nIters
print(trainError)

--Saving the trained model
paths = require 'paths'
filename  = paths.concat(paths.cwd(),'Frame.net')
torch.save(filename, rnn)

--Evaluating the trained model
rnn:evaluate()
correct=0
iters = 1
while iters<=nPredict do
  for i=1 ,133 do
    start = {}
    v = math.ceil(math.random()*2700)
    s = (i-1)*2800 + v
    table.insert(start,s)
    start = torch.Tensor(start)
    if gpu>0 then
      start=start:cuda()
    end
    inputs_test, targets_test = {}, {}
    for step=1,rho do
      inputs_test[step] = test_data:index(1, start):view(1,inputSize)
      targets_test[step] = test_labels:index(1, start):view(1,1)
      start:add(1)
    end

    rnn:zeroGradParameters()
    outputs_test = rnn:forward(inputs_test)
    _,position = torch.max(outputs_test[rho],2)
    
    a=targets_test[rho]:sum()
    b=position:sum()
    if (a == b) then
      correct = correct +1
    end
  end
  iters = iters+1
end
accuracy = ((correct*100)/(nPredict*outputSize))
print (accuracy)
print(correct)



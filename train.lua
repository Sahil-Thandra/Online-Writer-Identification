require('dpnn')
require('rnn')
require('nngraph')
require('optim')
data = require('data')

gpu=1
if gpu>0 then
  print("CUDA ON")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(gpu)
end

--Parameters
nIters = 3990
batchSize = 16   
rho = 100
hiddenSize = 800
inputSize = 3
outputSize = 133
lr = 0.001

--Loading the training data
train_data = data.traindataset()
train_labels = data.trainlabels()

if gpu>0 then
  train_data = train_data:cuda()
  train_labels = train_labels:cuda() 
end

--Loading the pre-trained model
paths = require 'paths'
filename  = paths.concat(paths.cwd(),'Train3.net')
model = torch.load(filename)

print(model)

if gpu>0 then
  model=model:cuda()
end

--Acquiring the model parameters
paramx,paramdx = model:getParameters()

--Loss Function
criterion =nn.CrossEntropyCriterion()

if gpu>0 then
  criterion=criterion:cuda()
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
    
    model:training()
    model:zeroGradParameters()
    outputs_table = model:forward(inputs)
    outputs = table2Tensor(outputs_table)
    targets_tensor = table2Tensor(targets)
    if gpu>0 then
      outputs = outputs:cuda()
      targets_tensor = targets_tensor:cuda()
    end
    err = criterion:forward(outputs[rho], targets_tensor[rho])
    gradOutputs = criterion:backward(outputs[rho], targets_tensor[rho])
    gradOutputsZeroed[rho] = gradOutputs
    model:backward(inputs, gradOutputsZeroed)
  return err, paramdx
end   

filename  = paths.concat(paths.cwd(),'Train.net')

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

    if iteration == 1 then
      min = Error[1]
      torch.save(filename,model)
    end
    
    if Error[1] < min then
      min = Error[1]
      torch.save(filename,model)  
    end

   iteration = iteration + 1
end
trainError = trainError/nIters
print(trainError)
print(min)

--Saving the trained model
paths = require 'paths'
filename1  = paths.concat(paths.cwd(),'Train6.net')
torch.save(filename1, model)


package.path = "../?.lua;" .. package.path
require 'torch';
require 'math'
require 'io'
require 'cutorch';
require 'nn';
require 'cunn';
require 'nngraph'
require 'image'
require 'NormCrossMapSSIM'
require 'NormCrossMapCorrelation'

local logger = require 'log'
logger.outfile = 'NormCrossMapSSIM.log'
-----------------------------------------------------------
--function to customize memory type
--
opt = {}

local function customizeDouble(memory)
  return memory:double()
end

local function customizeForCuda(memory)
  return memory:cuda()
end
--------------------------------------------------------------
-- INPUT SETTINGS
--
 
LAYERS = 6
INPUT_ROWS = 20
INPUT_COLUMNS = 32
STRIDE = 5
PATCHSIZE = 5
VERTICALWIDTH = 5
TOTAL_PATCH_CELLS = PATCHSIZE * PATCHSIZE
OUTPUT_ROWS = math.floor(INPUT_ROWS/STRIDE) + 1
OUTPUT_COLUMNS = math.floor(INPUT_COLUMNS/STRIDE) + 1
IsVerifyFlowOutput = true

tensor1= torch.Tensor(LAYERS/2, INPUT_ROWS, INPUT_COLUMNS):rand(LAYERS/2, INPUT_ROWS, INPUT_COLUMNS) * 10 --torch.range(1,25):reshape(1, 5,5)--image.load('1.png')--
tensor2= torch.Tensor(LAYERS/2, INPUT_ROWS, INPUT_COLUMNS):rand(LAYERS/2, INPUT_ROWS, INPUT_COLUMNS) * 10 --torch.range(1,25):reshape(1,5,5)--image.load('2.png')--
tensor1 = customizeDouble(tensor1);
tensor2 = customizeDouble(tensor2);

dataDouble = {
    inputs = {tensor1, tensor2},
    targets = customizeDouble(torch.Tensor({1, 0}))
}

dataCuda = {
    inputs = {tensor1:cuda(), tensor2:cuda()},
    targets = customizeForCuda(torch.Tensor({1}))
}

writeFile = function(filename, data)
    -- Opens a file in append mode
    file = io.open(filename, "w")
    for index = 1, data:size(1) do
        number = data[index]
        -- appends a word test to the last line of the file
        file:write(string.format('%.4f\n', number))
    end
    -- closes the open file
    file:close(file) 
end

--function create_model()

------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------
  local n_classes = 2

  -- MODEL:  
  conv1 = nn.SpatialConvolutionMM(LAYERS/2, LAYERS/2, 3, 3, 1, 1, 1, 1)()
  conv2 = nn.SpatialConvolutionMM(LAYERS/2, LAYERS/2, 3, 3, 1, 1, 1, 1)()
  join = nn.JoinTable(1)({conv1, conv2})
  xcorrGPU = nn.NormCrossMapCorrelation(PATCHSIZE, VERTICALWIDTH)(join);
  conv3= nn.SpatialConvolutionMM((LAYERS/2) * (INPUT_COLUMNS * VERTICALWIDTH), 1, 3, 3, 1, 1, 1, 1)(xcorrGPU)
  reshape = nn.Reshape((INPUT_ROWS * INPUT_COLUMNS), false)(conv3)
  lin = nn.Linear((INPUT_ROWS * INPUT_COLUMNS), n_classes)(reshape)
  softoutput = nn.LogSoftMax()(lin)
  local modelGPU = nn.gModule({conv1, conv2}, {softoutput})
  modelGPU = customizeForCuda(modelGPU)
  
--  return modelNonGPU, modelGPU
--end

------------------------------------------------------------------------------
-- LOSS FUNCTION
------------------------------------------------------------------------------

--modelNonGPU, modelGPU = create_model()

--now forward the new data
--for i = 1, 20 do


local x = os.clock()
local criterion = customizeForCuda(nn.ClassNLLCriterion())

model = modelGPU;
input = dataCuda.inputs;
data = dataCuda;
customize = customizeForCuda;
parameters,gradParameters = model:getParameters()

--do return end
--io.read()
--customize(criterion)
--gradoutput = torch.Tensor(outputs2:size()):fill(1)

--------------------------------------------------------------------------------
-- function that numerically checks gradient of the loss:
-- f is the scalar-valued function
-- g returns the true gradient (assumes input to f is a 1d tensor)
-- returns difference, true gradient, and estimated gradient
local function checkgrad(f, g, x, eps)
  -- compute true gradient
  local grad = customize(g(x))
  
  -- compute numeric approximations to gradient
  local grad_est = customize(torch.Tensor(grad:size()))
  grad_est:zero()
  eps = 1e-4
  
  for i = 1, grad:size(1) do
    -- do something with x[i] and evaluate f twice, and put your estimate of df/dx_i into grad_est[i]
    xlua.progress(i, grad:size(1))
    
    --create a temporary tensor for X to hold the 'eps' in the appropriate position 
    tempX = customize(torch.Tensor(grad:size(1)))
    --print(torch.typename(tempX))
    --print(torch.typename(x))
    tempX:zero()
    tempX[i] = eps
    
    -- calculate delta parameters for gradient calculation
    x_plus_eps = x + tempX
    x_minus_eps = x - tempX
    
    -- by using delta set of parameters, estimate the gradient for particular parameter 
    gradient = (f(x_plus_eps) - f(x_minus_eps)) / (eps * 2);
    grad_est[i] = gradient
  end

  -- computes (symmetric) relative error of gradient
  local diff = torch.norm(grad - grad_est) / (2 * torch.norm(grad + grad_est))
  return diff, grad, grad_est
end

---------------------------------------------------------------------------
-- returns loss(params)
local f = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  return criterion:forward(model:forward(input), data.targets)
end

--------------------------------------------------------------------------
-- returns dloss(params)/dparams
local g = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  
  gradParameters:zero()

  local outputs = model:forward(input)
  criterion:forward(outputs, data.targets)
  model:backward(input, criterion:backward(outputs, data.targets))

  return gradParameters
end

--------------------------------------------------------------------

print 'checking gradient ...'

-- call the checkgrad function to get the actual and estimate of the gradient 
local diff, grad, est = checkgrad(f, g, parameters)

-- print the actual gradient from the predefined criterion
print('actual gradient : \n')
writeFile('actual_grad.txt', grad)  
print('actual gradient writing completed')

-- print the estimated gradient from the approximation method
print('estimated gradient : \n')
writeFile('estimate_grad.txt', est)

--variables to find cosine similarity 
nominator = torch.sum(torch.cmul(grad, est))
denominator =  ((torch.norm(grad)) * torch.norm(est))

local cosineSimilarity = nominator / denominator

--print the status to console
print('symmetric relative error : ' .. diff .. ' --> cosine similarity : ' .. cosineSimilarity..'\n\n')

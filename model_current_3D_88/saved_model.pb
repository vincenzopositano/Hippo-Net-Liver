??,
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??&
?
conv3d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *!
shared_nameconv3d_15/kernel
?
$conv3d_15/kernel/Read/ReadVariableOpReadVariableOpconv3d_15/kernel**
_output_shapes
:
 *
dtype0
t
conv3d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_15/bias
m
"conv3d_15/bias/Read/ReadVariableOpReadVariableOpconv3d_15/bias*
_output_shapes
: *
dtype0
?
batch_normalization_40/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_40/gamma
?
0batch_normalization_40/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_40/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_40/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_40/beta
?
/batch_normalization_40/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_40/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_40/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_40/moving_mean
?
6batch_normalization_40/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_40/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_40/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_40/moving_variance
?
:batch_normalization_40/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_40/moving_variance*
_output_shapes
: *
dtype0
?
conv3d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *!
shared_nameconv3d_16/kernel
?
$conv3d_16/kernel/Read/ReadVariableOpReadVariableOpconv3d_16/kernel**
_output_shapes
:
  *
dtype0
t
conv3d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_16/bias
m
"conv3d_16/bias/Read/ReadVariableOpReadVariableOpconv3d_16/bias*
_output_shapes
: *
dtype0
?
batch_normalization_41/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_41/gamma
?
0batch_normalization_41/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_41/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_41/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_41/beta
?
/batch_normalization_41/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_41/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_41/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_41/moving_mean
?
6batch_normalization_41/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_41/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_41/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_41/moving_variance
?
:batch_normalization_41/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_41/moving_variance*
_output_shapes
: *
dtype0
?
conv3d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *!
shared_nameconv3d_17/kernel
?
$conv3d_17/kernel/Read/ReadVariableOpReadVariableOpconv3d_17/kernel**
_output_shapes
:
  *
dtype0
t
conv3d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_17/bias
m
"conv3d_17/bias/Read/ReadVariableOpReadVariableOpconv3d_17/bias*
_output_shapes
: *
dtype0
?
batch_normalization_42/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_42/gamma
?
0batch_normalization_42/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_42/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_42/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_42/beta
?
/batch_normalization_42/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_42/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_42/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_42/moving_mean
?
6batch_normalization_42/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_42/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_42/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_42/moving_variance
?
:batch_normalization_42/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_42/moving_variance*
_output_shapes
: *
dtype0
?
conv3d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *!
shared_nameconv3d_18/kernel
?
$conv3d_18/kernel/Read/ReadVariableOpReadVariableOpconv3d_18/kernel**
_output_shapes
:
  *
dtype0
t
conv3d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_18/bias
m
"conv3d_18/bias/Read/ReadVariableOpReadVariableOpconv3d_18/bias*
_output_shapes
: *
dtype0
?
batch_normalization_43/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_43/gamma
?
0batch_normalization_43/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_43/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_43/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_43/beta
?
/batch_normalization_43/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_43/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_43/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_43/moving_mean
?
6batch_normalization_43/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_43/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_43/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_43/moving_variance
?
:batch_normalization_43/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_43/moving_variance*
_output_shapes
: *
dtype0
?
conv3d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *!
shared_nameconv3d_19/kernel
?
$conv3d_19/kernel/Read/ReadVariableOpReadVariableOpconv3d_19/kernel**
_output_shapes
:
  *
dtype0
t
conv3d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_19/bias
m
"conv3d_19/bias/Read/ReadVariableOpReadVariableOpconv3d_19/bias*
_output_shapes
: *
dtype0
?
batch_normalization_44/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_44/gamma
?
0batch_normalization_44/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_44/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_44/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_44/beta
?
/batch_normalization_44/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_44/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_44/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_44/moving_mean
?
6batch_normalization_44/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_44/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_44/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_44/moving_variance
?
:batch_normalization_44/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_44/moving_variance*
_output_shapes
: *
dtype0
|
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?
?* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
?
?*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:?*
dtype0
{
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_17/kernel
t
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes
:	?*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv3d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *(
shared_nameAdam/conv3d_15/kernel/m
?
+Adam/conv3d_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/kernel/m**
_output_shapes
:
 *
dtype0
?
Adam/conv3d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_15/bias/m
{
)Adam/conv3d_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_40/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_40/gamma/m
?
7Adam/batch_normalization_40/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_40/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_40/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_40/beta/m
?
6Adam/batch_normalization_40/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_40/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv3d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *(
shared_nameAdam/conv3d_16/kernel/m
?
+Adam/conv3d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/kernel/m**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_16/bias/m
{
)Adam/conv3d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_41/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_41/gamma/m
?
7Adam/batch_normalization_41/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_41/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_41/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_41/beta/m
?
6Adam/batch_normalization_41/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_41/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv3d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *(
shared_nameAdam/conv3d_17/kernel/m
?
+Adam/conv3d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/kernel/m**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_17/bias/m
{
)Adam/conv3d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_42/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_42/gamma/m
?
7Adam/batch_normalization_42/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_42/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_42/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_42/beta/m
?
6Adam/batch_normalization_42/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_42/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv3d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *(
shared_nameAdam/conv3d_18/kernel/m
?
+Adam/conv3d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/kernel/m**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_18/bias/m
{
)Adam/conv3d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_43/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_43/gamma/m
?
7Adam/batch_normalization_43/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_43/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_43/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_43/beta/m
?
6Adam/batch_normalization_43/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_43/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv3d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *(
shared_nameAdam/conv3d_19/kernel/m
?
+Adam/conv3d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/kernel/m**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_19/bias/m
{
)Adam/conv3d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_44/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_44/gamma/m
?
7Adam/batch_normalization_44/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_44/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_44/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_44/beta/m
?
6Adam/batch_normalization_44/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_44/beta/m*
_output_shapes
: *
dtype0
?
Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?
?*'
shared_nameAdam/dense_16/kernel/m
?
*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m* 
_output_shapes
:
?
?*
dtype0
?
Adam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_16/bias/m
z
(Adam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_17/kernel/m
?
*Adam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/m
y
(Adam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *(
shared_nameAdam/conv3d_15/kernel/v
?
+Adam/conv3d_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/kernel/v**
_output_shapes
:
 *
dtype0
?
Adam/conv3d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_15/bias/v
{
)Adam/conv3d_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_40/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_40/gamma/v
?
7Adam/batch_normalization_40/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_40/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_40/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_40/beta/v
?
6Adam/batch_normalization_40/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_40/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv3d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *(
shared_nameAdam/conv3d_16/kernel/v
?
+Adam/conv3d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/kernel/v**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_16/bias/v
{
)Adam/conv3d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_41/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_41/gamma/v
?
7Adam/batch_normalization_41/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_41/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_41/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_41/beta/v
?
6Adam/batch_normalization_41/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_41/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv3d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *(
shared_nameAdam/conv3d_17/kernel/v
?
+Adam/conv3d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/kernel/v**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_17/bias/v
{
)Adam/conv3d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_42/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_42/gamma/v
?
7Adam/batch_normalization_42/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_42/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_42/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_42/beta/v
?
6Adam/batch_normalization_42/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_42/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv3d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *(
shared_nameAdam/conv3d_18/kernel/v
?
+Adam/conv3d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/kernel/v**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_18/bias/v
{
)Adam/conv3d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_43/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_43/gamma/v
?
7Adam/batch_normalization_43/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_43/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_43/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_43/beta/v
?
6Adam/batch_normalization_43/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_43/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv3d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *(
shared_nameAdam/conv3d_19/kernel/v
?
+Adam/conv3d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/kernel/v**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_19/bias/v
{
)Adam/conv3d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_44/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_44/gamma/v
?
7Adam/batch_normalization_44/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_44/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_44/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_44/beta/v
?
6Adam/batch_normalization_44/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_44/beta/v*
_output_shapes
: *
dtype0
?
Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?
?*'
shared_nameAdam/dense_16/kernel/v
?
*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v* 
_output_shapes
:
?
?*
dtype0
?
Adam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_16/bias/v
z
(Adam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_17/kernel/v
?
*Adam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/v
y
(Adam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv3d_15/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *+
shared_nameAdam/conv3d_15/kernel/vhat
?
.Adam/conv3d_15/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/kernel/vhat**
_output_shapes
:
 *
dtype0
?
Adam/conv3d_15/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv3d_15/bias/vhat
?
,Adam/conv3d_15/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/bias/vhat*
_output_shapes
: *
dtype0
?
&Adam/batch_normalization_40/gamma/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/batch_normalization_40/gamma/vhat
?
:Adam/batch_normalization_40/gamma/vhat/Read/ReadVariableOpReadVariableOp&Adam/batch_normalization_40/gamma/vhat*
_output_shapes
: *
dtype0
?
%Adam/batch_normalization_40/beta/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/batch_normalization_40/beta/vhat
?
9Adam/batch_normalization_40/beta/vhat/Read/ReadVariableOpReadVariableOp%Adam/batch_normalization_40/beta/vhat*
_output_shapes
: *
dtype0
?
Adam/conv3d_16/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *+
shared_nameAdam/conv3d_16/kernel/vhat
?
.Adam/conv3d_16/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/kernel/vhat**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_16/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv3d_16/bias/vhat
?
,Adam/conv3d_16/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/bias/vhat*
_output_shapes
: *
dtype0
?
&Adam/batch_normalization_41/gamma/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/batch_normalization_41/gamma/vhat
?
:Adam/batch_normalization_41/gamma/vhat/Read/ReadVariableOpReadVariableOp&Adam/batch_normalization_41/gamma/vhat*
_output_shapes
: *
dtype0
?
%Adam/batch_normalization_41/beta/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/batch_normalization_41/beta/vhat
?
9Adam/batch_normalization_41/beta/vhat/Read/ReadVariableOpReadVariableOp%Adam/batch_normalization_41/beta/vhat*
_output_shapes
: *
dtype0
?
Adam/conv3d_17/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *+
shared_nameAdam/conv3d_17/kernel/vhat
?
.Adam/conv3d_17/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/kernel/vhat**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_17/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv3d_17/bias/vhat
?
,Adam/conv3d_17/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/bias/vhat*
_output_shapes
: *
dtype0
?
&Adam/batch_normalization_42/gamma/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/batch_normalization_42/gamma/vhat
?
:Adam/batch_normalization_42/gamma/vhat/Read/ReadVariableOpReadVariableOp&Adam/batch_normalization_42/gamma/vhat*
_output_shapes
: *
dtype0
?
%Adam/batch_normalization_42/beta/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/batch_normalization_42/beta/vhat
?
9Adam/batch_normalization_42/beta/vhat/Read/ReadVariableOpReadVariableOp%Adam/batch_normalization_42/beta/vhat*
_output_shapes
: *
dtype0
?
Adam/conv3d_18/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *+
shared_nameAdam/conv3d_18/kernel/vhat
?
.Adam/conv3d_18/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/kernel/vhat**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_18/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv3d_18/bias/vhat
?
,Adam/conv3d_18/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_18/bias/vhat*
_output_shapes
: *
dtype0
?
&Adam/batch_normalization_43/gamma/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/batch_normalization_43/gamma/vhat
?
:Adam/batch_normalization_43/gamma/vhat/Read/ReadVariableOpReadVariableOp&Adam/batch_normalization_43/gamma/vhat*
_output_shapes
: *
dtype0
?
%Adam/batch_normalization_43/beta/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/batch_normalization_43/beta/vhat
?
9Adam/batch_normalization_43/beta/vhat/Read/ReadVariableOpReadVariableOp%Adam/batch_normalization_43/beta/vhat*
_output_shapes
: *
dtype0
?
Adam/conv3d_19/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *+
shared_nameAdam/conv3d_19/kernel/vhat
?
.Adam/conv3d_19/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/kernel/vhat**
_output_shapes
:
  *
dtype0
?
Adam/conv3d_19/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv3d_19/bias/vhat
?
,Adam/conv3d_19/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_19/bias/vhat*
_output_shapes
: *
dtype0
?
&Adam/batch_normalization_44/gamma/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/batch_normalization_44/gamma/vhat
?
:Adam/batch_normalization_44/gamma/vhat/Read/ReadVariableOpReadVariableOp&Adam/batch_normalization_44/gamma/vhat*
_output_shapes
: *
dtype0
?
%Adam/batch_normalization_44/beta/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/batch_normalization_44/beta/vhat
?
9Adam/batch_normalization_44/beta/vhat/Read/ReadVariableOpReadVariableOp%Adam/batch_normalization_44/beta/vhat*
_output_shapes
: *
dtype0
?
Adam/dense_16/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?
?**
shared_nameAdam/dense_16/kernel/vhat
?
-Adam/dense_16/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/vhat* 
_output_shapes
:
?
?*
dtype0
?
Adam/dense_16/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/dense_16/bias/vhat
?
+Adam/dense_16/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/vhat*
_output_shapes	
:?*
dtype0
?
Adam/dense_17/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameAdam/dense_17/kernel/vhat
?
-Adam/dense_17/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/vhat*
_output_shapes
:	?*
dtype0
?
Adam/dense_17/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dense_17/bias/vhat

+Adam/dense_17/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/vhat*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer-15
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
?
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&	variables
'regularization_losses
(trainable_variables
)	keras_api
R
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
?
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:regularization_losses
;trainable_variables
<	keras_api
R
=	variables
>regularization_losses
?trainable_variables
@	keras_api
h

Akernel
Bbias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
?
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
R
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
h

Tkernel
Ubias
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_	variables
`regularization_losses
atrainable_variables
b	keras_api
R
c	variables
dregularization_losses
etrainable_variables
f	keras_api
h

gkernel
hbias
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
?
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
R
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
R
z	variables
{regularization_losses
|trainable_variables
}	keras_api
T
~	variables
regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?"m?#m?.m?/m?5m?6m?Am?Bm?Hm?Im?Tm?Um?[m?\m?gm?hm?nm?om?	?m?	?m?	?m?	?m?v?v?"v?#v?.v?/v?5v?6v?Av?Bv?Hv?Iv?Tv?Uv?[v?\v?gv?hv?nv?ov?	?v?	?v?	?v?	?v?vhat?vhat?"vhat?#vhat?.vhat?/vhat?5vhat?6vhat?Avhat?Bvhat?Hvhat?Ivhat?Tvhat?Uvhat?[vhat?\vhat?gvhat?hvhat?nvhat?ovhat??vhat??vhat??vhat??vhat?
?
0
1
"2
#3
$4
%5
.6
/7
58
69
710
811
A12
B13
H14
I15
J16
K17
T18
U19
[20
\21
]22
^23
g24
h25
n26
o27
p28
q29
?30
?31
?32
?33
 
?
0
1
"2
#3
.4
/5
56
67
A8
B9
H10
I11
T12
U13
[14
\15
g16
h17
n18
o19
?20
?21
?22
?23
?
?non_trainable_variables
	variables
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
trainable_variables
 
\Z
VARIABLE_VALUEconv3d_15/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_15/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
?non_trainable_variables
	variables
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_40/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_40/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_40/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_40/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
$2
%3
 

"0
#1
?
?non_trainable_variables
&	variables
'regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
(trainable_variables
 
 
 
?
?non_trainable_variables
*	variables
+regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
,trainable_variables
\Z
VARIABLE_VALUEconv3d_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
?
?non_trainable_variables
0	variables
1regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
2trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_41/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_41/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_41/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_41/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

50
61
72
83
 

50
61
?
?non_trainable_variables
9	variables
:regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
;trainable_variables
 
 
 
?
?non_trainable_variables
=	variables
>regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
?trainable_variables
\Z
VARIABLE_VALUEconv3d_17/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_17/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
 

A0
B1
?
?non_trainable_variables
C	variables
Dregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
Etrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_42/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_42/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_42/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_42/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
J2
K3
 

H0
I1
?
?non_trainable_variables
L	variables
Mregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
Ntrainable_variables
 
 
 
?
?non_trainable_variables
P	variables
Qregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
Rtrainable_variables
\Z
VARIABLE_VALUEconv3d_18/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_18/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1
 

T0
U1
?
?non_trainable_variables
V	variables
Wregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
Xtrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_43/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_43/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_43/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_43/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

[0
\1
]2
^3
 

[0
\1
?
?non_trainable_variables
_	variables
`regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
atrainable_variables
 
 
 
?
?non_trainable_variables
c	variables
dregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
etrainable_variables
\Z
VARIABLE_VALUEconv3d_19/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_19/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1
 

g0
h1
?
?non_trainable_variables
i	variables
jregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
ktrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_44/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_44/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_44/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_44/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

n0
o1
p2
q3
 

n0
o1
?
?non_trainable_variables
r	variables
sregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
ttrainable_variables
 
 
 
?
?non_trainable_variables
v	variables
wregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
xtrainable_variables
 
 
 
?
?non_trainable_variables
z	variables
{regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
|trainable_variables
 
 
 
?
?non_trainable_variables
~	variables
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
?trainable_variables
\Z
VARIABLE_VALUEdense_16/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_16/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?non_trainable_variables
?	variables
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
?trainable_variables
 
 
 
?
?non_trainable_variables
?	variables
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
?trainable_variables
\Z
VARIABLE_VALUEdense_17/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_17/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?non_trainable_variables
?	variables
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
?trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
F
$0
%1
72
83
J4
K5
]6
^7
p8
q9

?0
?1
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
 
 
 
 
 
 

$0
%1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

70
81
 
 
 
 
 
 
 
 
 
 
 
 
 
 

J0
K1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

]0
^1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

p0
q1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv3d_15/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_15/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_40/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_40/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_16/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_16/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_41/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_41/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_17/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_17/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_42/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_42/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_18/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_18/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_43/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_43/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_19/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_19/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_44/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_44/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_16/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_16/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_17/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_17/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_15/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_15/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_40/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_40/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_16/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_16/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_41/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_41/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_17/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_17/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_42/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_42/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_18/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_18/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_43/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_43/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_19/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_19/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_44/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_44/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_16/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_16/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_17/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_17/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv3d_15/kernel/vhatUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv3d_15/bias/vhatSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/batch_normalization_40/gamma/vhatTlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%Adam/batch_normalization_40/beta/vhatSlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv3d_16/kernel/vhatUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv3d_16/bias/vhatSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/batch_normalization_41/gamma/vhatTlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%Adam/batch_normalization_41/beta/vhatSlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv3d_17/kernel/vhatUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv3d_17/bias/vhatSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/batch_normalization_42/gamma/vhatTlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%Adam/batch_normalization_42/beta/vhatSlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv3d_18/kernel/vhatUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv3d_18/bias/vhatSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/batch_normalization_43/gamma/vhatTlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%Adam/batch_normalization_43/beta/vhatSlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv3d_19/kernel/vhatUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv3d_19/bias/vhatSlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/batch_normalization_44/gamma/vhatTlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%Adam/batch_normalization_44/beta/vhatSlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_16/kernel/vhatVlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/dense_16/bias/vhatTlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_17/kernel/vhatVlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/dense_17/bias/vhatTlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv3d_15_inputPlaceholder*3
_output_shapes!
:?????????
@@*
dtype0*(
shape:?????????
@@
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv3d_15_inputconv3d_15/kernelconv3d_15/bias&batch_normalization_40/moving_variancebatch_normalization_40/gamma"batch_normalization_40/moving_meanbatch_normalization_40/betaconv3d_16/kernelconv3d_16/bias&batch_normalization_41/moving_variancebatch_normalization_41/gamma"batch_normalization_41/moving_meanbatch_normalization_41/betaconv3d_17/kernelconv3d_17/bias&batch_normalization_42/moving_variancebatch_normalization_42/gamma"batch_normalization_42/moving_meanbatch_normalization_42/betaconv3d_18/kernelconv3d_18/bias&batch_normalization_43/moving_variancebatch_normalization_43/gamma"batch_normalization_43/moving_meanbatch_normalization_43/betaconv3d_19/kernelconv3d_19/bias&batch_normalization_44/moving_variancebatch_normalization_44/gamma"batch_normalization_44/moving_meanbatch_normalization_44/betadense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_420515
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?-
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv3d_15/kernel/Read/ReadVariableOp"conv3d_15/bias/Read/ReadVariableOp0batch_normalization_40/gamma/Read/ReadVariableOp/batch_normalization_40/beta/Read/ReadVariableOp6batch_normalization_40/moving_mean/Read/ReadVariableOp:batch_normalization_40/moving_variance/Read/ReadVariableOp$conv3d_16/kernel/Read/ReadVariableOp"conv3d_16/bias/Read/ReadVariableOp0batch_normalization_41/gamma/Read/ReadVariableOp/batch_normalization_41/beta/Read/ReadVariableOp6batch_normalization_41/moving_mean/Read/ReadVariableOp:batch_normalization_41/moving_variance/Read/ReadVariableOp$conv3d_17/kernel/Read/ReadVariableOp"conv3d_17/bias/Read/ReadVariableOp0batch_normalization_42/gamma/Read/ReadVariableOp/batch_normalization_42/beta/Read/ReadVariableOp6batch_normalization_42/moving_mean/Read/ReadVariableOp:batch_normalization_42/moving_variance/Read/ReadVariableOp$conv3d_18/kernel/Read/ReadVariableOp"conv3d_18/bias/Read/ReadVariableOp0batch_normalization_43/gamma/Read/ReadVariableOp/batch_normalization_43/beta/Read/ReadVariableOp6batch_normalization_43/moving_mean/Read/ReadVariableOp:batch_normalization_43/moving_variance/Read/ReadVariableOp$conv3d_19/kernel/Read/ReadVariableOp"conv3d_19/bias/Read/ReadVariableOp0batch_normalization_44/gamma/Read/ReadVariableOp/batch_normalization_44/beta/Read/ReadVariableOp6batch_normalization_44/moving_mean/Read/ReadVariableOp:batch_normalization_44/moving_variance/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv3d_15/kernel/m/Read/ReadVariableOp)Adam/conv3d_15/bias/m/Read/ReadVariableOp7Adam/batch_normalization_40/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_40/beta/m/Read/ReadVariableOp+Adam/conv3d_16/kernel/m/Read/ReadVariableOp)Adam/conv3d_16/bias/m/Read/ReadVariableOp7Adam/batch_normalization_41/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_41/beta/m/Read/ReadVariableOp+Adam/conv3d_17/kernel/m/Read/ReadVariableOp)Adam/conv3d_17/bias/m/Read/ReadVariableOp7Adam/batch_normalization_42/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_42/beta/m/Read/ReadVariableOp+Adam/conv3d_18/kernel/m/Read/ReadVariableOp)Adam/conv3d_18/bias/m/Read/ReadVariableOp7Adam/batch_normalization_43/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_43/beta/m/Read/ReadVariableOp+Adam/conv3d_19/kernel/m/Read/ReadVariableOp)Adam/conv3d_19/bias/m/Read/ReadVariableOp7Adam/batch_normalization_44/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_44/beta/m/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp(Adam/dense_16/bias/m/Read/ReadVariableOp*Adam/dense_17/kernel/m/Read/ReadVariableOp(Adam/dense_17/bias/m/Read/ReadVariableOp+Adam/conv3d_15/kernel/v/Read/ReadVariableOp)Adam/conv3d_15/bias/v/Read/ReadVariableOp7Adam/batch_normalization_40/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_40/beta/v/Read/ReadVariableOp+Adam/conv3d_16/kernel/v/Read/ReadVariableOp)Adam/conv3d_16/bias/v/Read/ReadVariableOp7Adam/batch_normalization_41/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_41/beta/v/Read/ReadVariableOp+Adam/conv3d_17/kernel/v/Read/ReadVariableOp)Adam/conv3d_17/bias/v/Read/ReadVariableOp7Adam/batch_normalization_42/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_42/beta/v/Read/ReadVariableOp+Adam/conv3d_18/kernel/v/Read/ReadVariableOp)Adam/conv3d_18/bias/v/Read/ReadVariableOp7Adam/batch_normalization_43/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_43/beta/v/Read/ReadVariableOp+Adam/conv3d_19/kernel/v/Read/ReadVariableOp)Adam/conv3d_19/bias/v/Read/ReadVariableOp7Adam/batch_normalization_44/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_44/beta/v/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp(Adam/dense_16/bias/v/Read/ReadVariableOp*Adam/dense_17/kernel/v/Read/ReadVariableOp(Adam/dense_17/bias/v/Read/ReadVariableOp.Adam/conv3d_15/kernel/vhat/Read/ReadVariableOp,Adam/conv3d_15/bias/vhat/Read/ReadVariableOp:Adam/batch_normalization_40/gamma/vhat/Read/ReadVariableOp9Adam/batch_normalization_40/beta/vhat/Read/ReadVariableOp.Adam/conv3d_16/kernel/vhat/Read/ReadVariableOp,Adam/conv3d_16/bias/vhat/Read/ReadVariableOp:Adam/batch_normalization_41/gamma/vhat/Read/ReadVariableOp9Adam/batch_normalization_41/beta/vhat/Read/ReadVariableOp.Adam/conv3d_17/kernel/vhat/Read/ReadVariableOp,Adam/conv3d_17/bias/vhat/Read/ReadVariableOp:Adam/batch_normalization_42/gamma/vhat/Read/ReadVariableOp9Adam/batch_normalization_42/beta/vhat/Read/ReadVariableOp.Adam/conv3d_18/kernel/vhat/Read/ReadVariableOp,Adam/conv3d_18/bias/vhat/Read/ReadVariableOp:Adam/batch_normalization_43/gamma/vhat/Read/ReadVariableOp9Adam/batch_normalization_43/beta/vhat/Read/ReadVariableOp.Adam/conv3d_19/kernel/vhat/Read/ReadVariableOp,Adam/conv3d_19/bias/vhat/Read/ReadVariableOp:Adam/batch_normalization_44/gamma/vhat/Read/ReadVariableOp9Adam/batch_normalization_44/beta/vhat/Read/ReadVariableOp-Adam/dense_16/kernel/vhat/Read/ReadVariableOp+Adam/dense_16/bias/vhat/Read/ReadVariableOp-Adam/dense_17/kernel/vhat/Read/ReadVariableOp+Adam/dense_17/bias/vhat/Read/ReadVariableOpConst*?
Tiny
w2u	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_422542
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d_15/kernelconv3d_15/biasbatch_normalization_40/gammabatch_normalization_40/beta"batch_normalization_40/moving_mean&batch_normalization_40/moving_varianceconv3d_16/kernelconv3d_16/biasbatch_normalization_41/gammabatch_normalization_41/beta"batch_normalization_41/moving_mean&batch_normalization_41/moving_varianceconv3d_17/kernelconv3d_17/biasbatch_normalization_42/gammabatch_normalization_42/beta"batch_normalization_42/moving_mean&batch_normalization_42/moving_varianceconv3d_18/kernelconv3d_18/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv3d_19/kernelconv3d_19/biasbatch_normalization_44/gammabatch_normalization_44/beta"batch_normalization_44/moving_mean&batch_normalization_44/moving_variancedense_16/kerneldense_16/biasdense_17/kerneldense_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv3d_15/kernel/mAdam/conv3d_15/bias/m#Adam/batch_normalization_40/gamma/m"Adam/batch_normalization_40/beta/mAdam/conv3d_16/kernel/mAdam/conv3d_16/bias/m#Adam/batch_normalization_41/gamma/m"Adam/batch_normalization_41/beta/mAdam/conv3d_17/kernel/mAdam/conv3d_17/bias/m#Adam/batch_normalization_42/gamma/m"Adam/batch_normalization_42/beta/mAdam/conv3d_18/kernel/mAdam/conv3d_18/bias/m#Adam/batch_normalization_43/gamma/m"Adam/batch_normalization_43/beta/mAdam/conv3d_19/kernel/mAdam/conv3d_19/bias/m#Adam/batch_normalization_44/gamma/m"Adam/batch_normalization_44/beta/mAdam/dense_16/kernel/mAdam/dense_16/bias/mAdam/dense_17/kernel/mAdam/dense_17/bias/mAdam/conv3d_15/kernel/vAdam/conv3d_15/bias/v#Adam/batch_normalization_40/gamma/v"Adam/batch_normalization_40/beta/vAdam/conv3d_16/kernel/vAdam/conv3d_16/bias/v#Adam/batch_normalization_41/gamma/v"Adam/batch_normalization_41/beta/vAdam/conv3d_17/kernel/vAdam/conv3d_17/bias/v#Adam/batch_normalization_42/gamma/v"Adam/batch_normalization_42/beta/vAdam/conv3d_18/kernel/vAdam/conv3d_18/bias/v#Adam/batch_normalization_43/gamma/v"Adam/batch_normalization_43/beta/vAdam/conv3d_19/kernel/vAdam/conv3d_19/bias/v#Adam/batch_normalization_44/gamma/v"Adam/batch_normalization_44/beta/vAdam/dense_16/kernel/vAdam/dense_16/bias/vAdam/dense_17/kernel/vAdam/dense_17/bias/vAdam/conv3d_15/kernel/vhatAdam/conv3d_15/bias/vhat&Adam/batch_normalization_40/gamma/vhat%Adam/batch_normalization_40/beta/vhatAdam/conv3d_16/kernel/vhatAdam/conv3d_16/bias/vhat&Adam/batch_normalization_41/gamma/vhat%Adam/batch_normalization_41/beta/vhatAdam/conv3d_17/kernel/vhatAdam/conv3d_17/bias/vhat&Adam/batch_normalization_42/gamma/vhat%Adam/batch_normalization_42/beta/vhatAdam/conv3d_18/kernel/vhatAdam/conv3d_18/bias/vhat&Adam/batch_normalization_43/gamma/vhat%Adam/batch_normalization_43/beta/vhatAdam/conv3d_19/kernel/vhatAdam/conv3d_19/bias/vhat&Adam/batch_normalization_44/gamma/vhat%Adam/batch_normalization_44/beta/vhatAdam/dense_16/kernel/vhatAdam/dense_16/bias/vhatAdam/dense_17/kernel/vhatAdam/dense_17/bias/vhat*
Tinx
v2t*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_422897??"
?

?
D__inference_dense_17_layer_call_and_return_conditional_losses_419319

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
+__inference_dropout_17_layer_call_fn_422093

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_4194512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_16_layer_call_and_return_conditional_losses_419295

inputs2
matmul_readvariableop_resource:
?
?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?,
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421779

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_420515
conv3d_15_input%
unknown:
 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: '
	unknown_5:
  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: (

unknown_11:
  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: (

unknown_17:
  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: (

unknown_23:
  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29:
?
?

unknown_30:	?

unknown_31:	?

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_4181332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
3
_output_shapes!
:?????????
@@
)
_user_specified_nameconv3d_15_input
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_421937

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_40_layer_call_fn_421123

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4182172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_422063

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????
2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????
2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????
2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_43_layer_call_fn_421712

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_4192052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_420661

inputs%
unknown:
 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: '
	unknown_5:
  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: (

unknown_11:
  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: (

unknown_17:
  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: (

unknown_23:
  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29:
?
?

unknown_30:	?

unknown_31:	?

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
 !"*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_4200342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
@@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_419099

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
   2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
   2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
   : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
   
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_30_layer_call_fn_418307

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_30_layer_call_and_return_conditional_losses_4183012
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?,
?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421203

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?,
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_421971

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_418157

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv3d_15_layer_call_and_return_conditional_losses_421097

inputs<
conv3d_readvariableop_resource:
 -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
 *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
@@ *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
@@ 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
@@ 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????
@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????
@@
 
_user_specified_nameinputs
?
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_422051

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????
2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????
2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_34_layer_call_fn_419003

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_34_layer_call_and_return_conditional_losses_4189972
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_dense_17_layer_call_and_return_conditional_losses_422130

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_419258

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_30_layer_call_and_return_conditional_losses_418301

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_dense_17_layer_call_fn_422119

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_4193192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_44_layer_call_fn_421891

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_4189132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_422163Y
;conv3d_18_kernel_regularizer_square_readvariableop_resource:
  
identity??2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
2conv3d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv3d_18_kernel_regularizer_square_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_18/kernel/Regularizer/SquareSquare:conv3d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_18/kernel/Regularizer/Square?
"conv3d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_18/kernel/Regularizer/Const?
 conv3d_18/kernel/Regularizer/SumSum'conv3d_18/kernel/Regularizer/Square:y:0+conv3d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/Sum?
"conv3d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_18/kernel/Regularizer/mul/x?
 conv3d_18/kernel/Regularizer/mulMul+conv3d_18/kernel/Regularizer/mul/x:output:0)conv3d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/mul?
IdentityIdentity$conv3d_18/kernel/Regularizer/mul:z:03^conv3d_18/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2conv3d_18/kernel/Regularizer/Square/ReadVariableOp
?+
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_419612

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?,
?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_418391

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?+
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_419542

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
?
*__inference_conv3d_15_layer_call_fn_421086

inputs%
unknown:
 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_4190212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
@@
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_42_layer_call_fn_421494

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_4185052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_422110

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421607

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_44_layer_call_fn_421917

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_4195422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?,
?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_418565

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_420588

inputs%
unknown:
 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: '
	unknown_5:
  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: (

unknown_11:
  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: (

unknown_17:
  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: (

unknown_23:
  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29:
?
?

unknown_30:	?

unknown_31:	?

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_4193502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
@@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_418331

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?,
?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421587

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?,
?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421395

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_41_layer_call_fn_421315

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_4183912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421745

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_418853

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv3d_18_layer_call_fn_421656

inputs%
unknown:
  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_18_layer_call_and_return_conditional_losses_4191802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
 : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_418679

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?

?
D__inference_dense_16_layer_call_and_return_conditional_losses_422083

inputs2
matmul_readvariableop_resource:
?
?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_32_layer_call_and_return_conditional_losses_418649

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv3d_19_layer_call_and_return_conditional_losses_421865

inputs<
conv3d_readvariableop_resource:
  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
Relu?
2conv3d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_19/kernel/Regularizer/SquareSquare:conv3d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_19/kernel/Regularizer/Square?
"conv3d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_19/kernel/Regularizer/Const?
 conv3d_19/kernel/Regularizer/SumSum'conv3d_19/kernel/Regularizer/Square:y:0+conv3d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/Sum?
"conv3d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_19/kernel/Regularizer/mul/x?
 conv3d_19/kernel/Regularizer/mulMul+conv3d_19/kernel/Regularizer/mul/x:output:0)conv3d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp3^conv3d_19/kernel/Regularizer/Square/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2h
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2conv3d_19/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_42_layer_call_fn_421507

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_4185652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
G
+__inference_dropout_16_layer_call_fn_422041

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_4192822
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_419282

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????
2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????
2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?,
?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_418217

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv3d_18_layer_call_and_return_conditional_losses_421673

inputs<
conv3d_readvariableop_resource:
  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
Relu?
2conv3d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_18/kernel/Regularizer/SquareSquare:conv3d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_18/kernel/Regularizer/Square?
"conv3d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_18/kernel/Regularizer/Const?
 conv3d_18/kernel/Regularizer/SumSum'conv3d_18/kernel/Regularizer/Square:y:0+conv3d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/Sum?
"conv3d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_18/kernel/Regularizer/mul/x?
 conv3d_18/kernel/Regularizer/mulMul+conv3d_18/kernel/Regularizer/mul/x:output:0)conv3d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp3^conv3d_18/kernel/Regularizer/Square/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2h
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2conv3d_18/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421361

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_44_layer_call_fn_421904

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_4192582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
ɝ
?%
!__inference__wrapped_model_418133
conv3d_15_inputS
5sequential_9_conv3d_15_conv3d_readvariableop_resource:
 D
6sequential_9_conv3d_15_biasadd_readvariableop_resource: S
Esequential_9_batch_normalization_40_batchnorm_readvariableop_resource: W
Isequential_9_batch_normalization_40_batchnorm_mul_readvariableop_resource: U
Gsequential_9_batch_normalization_40_batchnorm_readvariableop_1_resource: U
Gsequential_9_batch_normalization_40_batchnorm_readvariableop_2_resource: S
5sequential_9_conv3d_16_conv3d_readvariableop_resource:
  D
6sequential_9_conv3d_16_biasadd_readvariableop_resource: S
Esequential_9_batch_normalization_41_batchnorm_readvariableop_resource: W
Isequential_9_batch_normalization_41_batchnorm_mul_readvariableop_resource: U
Gsequential_9_batch_normalization_41_batchnorm_readvariableop_1_resource: U
Gsequential_9_batch_normalization_41_batchnorm_readvariableop_2_resource: S
5sequential_9_conv3d_17_conv3d_readvariableop_resource:
  D
6sequential_9_conv3d_17_biasadd_readvariableop_resource: S
Esequential_9_batch_normalization_42_batchnorm_readvariableop_resource: W
Isequential_9_batch_normalization_42_batchnorm_mul_readvariableop_resource: U
Gsequential_9_batch_normalization_42_batchnorm_readvariableop_1_resource: U
Gsequential_9_batch_normalization_42_batchnorm_readvariableop_2_resource: S
5sequential_9_conv3d_18_conv3d_readvariableop_resource:
  D
6sequential_9_conv3d_18_biasadd_readvariableop_resource: S
Esequential_9_batch_normalization_43_batchnorm_readvariableop_resource: W
Isequential_9_batch_normalization_43_batchnorm_mul_readvariableop_resource: U
Gsequential_9_batch_normalization_43_batchnorm_readvariableop_1_resource: U
Gsequential_9_batch_normalization_43_batchnorm_readvariableop_2_resource: S
5sequential_9_conv3d_19_conv3d_readvariableop_resource:
  D
6sequential_9_conv3d_19_biasadd_readvariableop_resource: S
Esequential_9_batch_normalization_44_batchnorm_readvariableop_resource: W
Isequential_9_batch_normalization_44_batchnorm_mul_readvariableop_resource: U
Gsequential_9_batch_normalization_44_batchnorm_readvariableop_1_resource: U
Gsequential_9_batch_normalization_44_batchnorm_readvariableop_2_resource: H
4sequential_9_dense_16_matmul_readvariableop_resource:
?
?D
5sequential_9_dense_16_biasadd_readvariableop_resource:	?G
4sequential_9_dense_17_matmul_readvariableop_resource:	?C
5sequential_9_dense_17_biasadd_readvariableop_resource:
identity??<sequential_9/batch_normalization_40/batchnorm/ReadVariableOp?>sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_1?>sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_2?@sequential_9/batch_normalization_40/batchnorm/mul/ReadVariableOp?<sequential_9/batch_normalization_41/batchnorm/ReadVariableOp?>sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_1?>sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_2?@sequential_9/batch_normalization_41/batchnorm/mul/ReadVariableOp?<sequential_9/batch_normalization_42/batchnorm/ReadVariableOp?>sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_1?>sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_2?@sequential_9/batch_normalization_42/batchnorm/mul/ReadVariableOp?<sequential_9/batch_normalization_43/batchnorm/ReadVariableOp?>sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_1?>sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_2?@sequential_9/batch_normalization_43/batchnorm/mul/ReadVariableOp?<sequential_9/batch_normalization_44/batchnorm/ReadVariableOp?>sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_1?>sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_2?@sequential_9/batch_normalization_44/batchnorm/mul/ReadVariableOp?-sequential_9/conv3d_15/BiasAdd/ReadVariableOp?,sequential_9/conv3d_15/Conv3D/ReadVariableOp?-sequential_9/conv3d_16/BiasAdd/ReadVariableOp?,sequential_9/conv3d_16/Conv3D/ReadVariableOp?-sequential_9/conv3d_17/BiasAdd/ReadVariableOp?,sequential_9/conv3d_17/Conv3D/ReadVariableOp?-sequential_9/conv3d_18/BiasAdd/ReadVariableOp?,sequential_9/conv3d_18/Conv3D/ReadVariableOp?-sequential_9/conv3d_19/BiasAdd/ReadVariableOp?,sequential_9/conv3d_19/Conv3D/ReadVariableOp?,sequential_9/dense_16/BiasAdd/ReadVariableOp?+sequential_9/dense_16/MatMul/ReadVariableOp?,sequential_9/dense_17/BiasAdd/ReadVariableOp?+sequential_9/dense_17/MatMul/ReadVariableOp?
,sequential_9/conv3d_15/Conv3D/ReadVariableOpReadVariableOp5sequential_9_conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:
 *
dtype02.
,sequential_9/conv3d_15/Conv3D/ReadVariableOp?
sequential_9/conv3d_15/Conv3DConv3Dconv3d_15_input4sequential_9/conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
@@ *
paddingSAME*
strides	
2
sequential_9/conv3d_15/Conv3D?
-sequential_9/conv3d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv3d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv3d_15/BiasAdd/ReadVariableOp?
sequential_9/conv3d_15/BiasAddBiasAdd&sequential_9/conv3d_15/Conv3D:output:05sequential_9/conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
@@ 2 
sequential_9/conv3d_15/BiasAdd?
sequential_9/conv3d_15/ReluRelu'sequential_9/conv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
@@ 2
sequential_9/conv3d_15/Relu?
<sequential_9/batch_normalization_40/batchnorm/ReadVariableOpReadVariableOpEsequential_9_batch_normalization_40_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02>
<sequential_9/batch_normalization_40/batchnorm/ReadVariableOp?
3sequential_9/batch_normalization_40/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_9/batch_normalization_40/batchnorm/add/y?
1sequential_9/batch_normalization_40/batchnorm/addAddV2Dsequential_9/batch_normalization_40/batchnorm/ReadVariableOp:value:0<sequential_9/batch_normalization_40/batchnorm/add/y:output:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_40/batchnorm/add?
3sequential_9/batch_normalization_40/batchnorm/RsqrtRsqrt5sequential_9/batch_normalization_40/batchnorm/add:z:0*
T0*
_output_shapes
: 25
3sequential_9/batch_normalization_40/batchnorm/Rsqrt?
@sequential_9/batch_normalization_40/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_9_batch_normalization_40_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02B
@sequential_9/batch_normalization_40/batchnorm/mul/ReadVariableOp?
1sequential_9/batch_normalization_40/batchnorm/mulMul7sequential_9/batch_normalization_40/batchnorm/Rsqrt:y:0Hsequential_9/batch_normalization_40/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_40/batchnorm/mul?
3sequential_9/batch_normalization_40/batchnorm/mul_1Mul)sequential_9/conv3d_15/Relu:activations:05sequential_9/batch_normalization_40/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
@@ 25
3sequential_9/batch_normalization_40/batchnorm/mul_1?
>sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_9_batch_normalization_40_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02@
>sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_1?
3sequential_9/batch_normalization_40/batchnorm/mul_2MulFsequential_9/batch_normalization_40/batchnorm/ReadVariableOp_1:value:05sequential_9/batch_normalization_40/batchnorm/mul:z:0*
T0*
_output_shapes
: 25
3sequential_9/batch_normalization_40/batchnorm/mul_2?
>sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_9_batch_normalization_40_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02@
>sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_2?
1sequential_9/batch_normalization_40/batchnorm/subSubFsequential_9/batch_normalization_40/batchnorm/ReadVariableOp_2:value:07sequential_9/batch_normalization_40/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_40/batchnorm/sub?
3sequential_9/batch_normalization_40/batchnorm/add_1AddV27sequential_9/batch_normalization_40/batchnorm/mul_1:z:05sequential_9/batch_normalization_40/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
@@ 25
3sequential_9/batch_normalization_40/batchnorm/add_1?
'sequential_9/max_pooling3d_30/MaxPool3D	MaxPool3D7sequential_9/batch_normalization_40/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
   *
ksize	
*
paddingVALID*
strides	
2)
'sequential_9/max_pooling3d_30/MaxPool3D?
,sequential_9/conv3d_16/Conv3D/ReadVariableOpReadVariableOp5sequential_9_conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02.
,sequential_9/conv3d_16/Conv3D/ReadVariableOp?
sequential_9/conv3d_16/Conv3DConv3D0sequential_9/max_pooling3d_30/MaxPool3D:output:04sequential_9/conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
   *
paddingSAME*
strides	
2
sequential_9/conv3d_16/Conv3D?
-sequential_9/conv3d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv3d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv3d_16/BiasAdd/ReadVariableOp?
sequential_9/conv3d_16/BiasAddBiasAdd&sequential_9/conv3d_16/Conv3D:output:05sequential_9/conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
   2 
sequential_9/conv3d_16/BiasAdd?
sequential_9/conv3d_16/ReluRelu'sequential_9/conv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
   2
sequential_9/conv3d_16/Relu?
<sequential_9/batch_normalization_41/batchnorm/ReadVariableOpReadVariableOpEsequential_9_batch_normalization_41_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02>
<sequential_9/batch_normalization_41/batchnorm/ReadVariableOp?
3sequential_9/batch_normalization_41/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_9/batch_normalization_41/batchnorm/add/y?
1sequential_9/batch_normalization_41/batchnorm/addAddV2Dsequential_9/batch_normalization_41/batchnorm/ReadVariableOp:value:0<sequential_9/batch_normalization_41/batchnorm/add/y:output:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_41/batchnorm/add?
3sequential_9/batch_normalization_41/batchnorm/RsqrtRsqrt5sequential_9/batch_normalization_41/batchnorm/add:z:0*
T0*
_output_shapes
: 25
3sequential_9/batch_normalization_41/batchnorm/Rsqrt?
@sequential_9/batch_normalization_41/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_9_batch_normalization_41_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02B
@sequential_9/batch_normalization_41/batchnorm/mul/ReadVariableOp?
1sequential_9/batch_normalization_41/batchnorm/mulMul7sequential_9/batch_normalization_41/batchnorm/Rsqrt:y:0Hsequential_9/batch_normalization_41/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_41/batchnorm/mul?
3sequential_9/batch_normalization_41/batchnorm/mul_1Mul)sequential_9/conv3d_16/Relu:activations:05sequential_9/batch_normalization_41/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
   25
3sequential_9/batch_normalization_41/batchnorm/mul_1?
>sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_9_batch_normalization_41_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02@
>sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_1?
3sequential_9/batch_normalization_41/batchnorm/mul_2MulFsequential_9/batch_normalization_41/batchnorm/ReadVariableOp_1:value:05sequential_9/batch_normalization_41/batchnorm/mul:z:0*
T0*
_output_shapes
: 25
3sequential_9/batch_normalization_41/batchnorm/mul_2?
>sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_9_batch_normalization_41_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02@
>sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_2?
1sequential_9/batch_normalization_41/batchnorm/subSubFsequential_9/batch_normalization_41/batchnorm/ReadVariableOp_2:value:07sequential_9/batch_normalization_41/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_41/batchnorm/sub?
3sequential_9/batch_normalization_41/batchnorm/add_1AddV27sequential_9/batch_normalization_41/batchnorm/mul_1:z:05sequential_9/batch_normalization_41/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
   25
3sequential_9/batch_normalization_41/batchnorm/add_1?
'sequential_9/max_pooling3d_31/MaxPool3D	MaxPool3D7sequential_9/batch_normalization_41/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2)
'sequential_9/max_pooling3d_31/MaxPool3D?
,sequential_9/conv3d_17/Conv3D/ReadVariableOpReadVariableOp5sequential_9_conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02.
,sequential_9/conv3d_17/Conv3D/ReadVariableOp?
sequential_9/conv3d_17/Conv3DConv3D0sequential_9/max_pooling3d_31/MaxPool3D:output:04sequential_9/conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
sequential_9/conv3d_17/Conv3D?
-sequential_9/conv3d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv3d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv3d_17/BiasAdd/ReadVariableOp?
sequential_9/conv3d_17/BiasAddBiasAdd&sequential_9/conv3d_17/Conv3D:output:05sequential_9/conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2 
sequential_9/conv3d_17/BiasAdd?
sequential_9/conv3d_17/ReluRelu'sequential_9/conv3d_17/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
sequential_9/conv3d_17/Relu?
<sequential_9/batch_normalization_42/batchnorm/ReadVariableOpReadVariableOpEsequential_9_batch_normalization_42_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02>
<sequential_9/batch_normalization_42/batchnorm/ReadVariableOp?
3sequential_9/batch_normalization_42/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_9/batch_normalization_42/batchnorm/add/y?
1sequential_9/batch_normalization_42/batchnorm/addAddV2Dsequential_9/batch_normalization_42/batchnorm/ReadVariableOp:value:0<sequential_9/batch_normalization_42/batchnorm/add/y:output:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_42/batchnorm/add?
3sequential_9/batch_normalization_42/batchnorm/RsqrtRsqrt5sequential_9/batch_normalization_42/batchnorm/add:z:0*
T0*
_output_shapes
: 25
3sequential_9/batch_normalization_42/batchnorm/Rsqrt?
@sequential_9/batch_normalization_42/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_9_batch_normalization_42_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02B
@sequential_9/batch_normalization_42/batchnorm/mul/ReadVariableOp?
1sequential_9/batch_normalization_42/batchnorm/mulMul7sequential_9/batch_normalization_42/batchnorm/Rsqrt:y:0Hsequential_9/batch_normalization_42/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_42/batchnorm/mul?
3sequential_9/batch_normalization_42/batchnorm/mul_1Mul)sequential_9/conv3d_17/Relu:activations:05sequential_9/batch_normalization_42/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 25
3sequential_9/batch_normalization_42/batchnorm/mul_1?
>sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_9_batch_normalization_42_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02@
>sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_1?
3sequential_9/batch_normalization_42/batchnorm/mul_2MulFsequential_9/batch_normalization_42/batchnorm/ReadVariableOp_1:value:05sequential_9/batch_normalization_42/batchnorm/mul:z:0*
T0*
_output_shapes
: 25
3sequential_9/batch_normalization_42/batchnorm/mul_2?
>sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_9_batch_normalization_42_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02@
>sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_2?
1sequential_9/batch_normalization_42/batchnorm/subSubFsequential_9/batch_normalization_42/batchnorm/ReadVariableOp_2:value:07sequential_9/batch_normalization_42/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_42/batchnorm/sub?
3sequential_9/batch_normalization_42/batchnorm/add_1AddV27sequential_9/batch_normalization_42/batchnorm/mul_1:z:05sequential_9/batch_normalization_42/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 25
3sequential_9/batch_normalization_42/batchnorm/add_1?
'sequential_9/max_pooling3d_32/MaxPool3D	MaxPool3D7sequential_9/batch_normalization_42/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2)
'sequential_9/max_pooling3d_32/MaxPool3D?
,sequential_9/conv3d_18/Conv3D/ReadVariableOpReadVariableOp5sequential_9_conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02.
,sequential_9/conv3d_18/Conv3D/ReadVariableOp?
sequential_9/conv3d_18/Conv3DConv3D0sequential_9/max_pooling3d_32/MaxPool3D:output:04sequential_9/conv3d_18/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
sequential_9/conv3d_18/Conv3D?
-sequential_9/conv3d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv3d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv3d_18/BiasAdd/ReadVariableOp?
sequential_9/conv3d_18/BiasAddBiasAdd&sequential_9/conv3d_18/Conv3D:output:05sequential_9/conv3d_18/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2 
sequential_9/conv3d_18/BiasAdd?
sequential_9/conv3d_18/ReluRelu'sequential_9/conv3d_18/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
sequential_9/conv3d_18/Relu?
<sequential_9/batch_normalization_43/batchnorm/ReadVariableOpReadVariableOpEsequential_9_batch_normalization_43_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02>
<sequential_9/batch_normalization_43/batchnorm/ReadVariableOp?
3sequential_9/batch_normalization_43/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_9/batch_normalization_43/batchnorm/add/y?
1sequential_9/batch_normalization_43/batchnorm/addAddV2Dsequential_9/batch_normalization_43/batchnorm/ReadVariableOp:value:0<sequential_9/batch_normalization_43/batchnorm/add/y:output:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_43/batchnorm/add?
3sequential_9/batch_normalization_43/batchnorm/RsqrtRsqrt5sequential_9/batch_normalization_43/batchnorm/add:z:0*
T0*
_output_shapes
: 25
3sequential_9/batch_normalization_43/batchnorm/Rsqrt?
@sequential_9/batch_normalization_43/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_9_batch_normalization_43_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02B
@sequential_9/batch_normalization_43/batchnorm/mul/ReadVariableOp?
1sequential_9/batch_normalization_43/batchnorm/mulMul7sequential_9/batch_normalization_43/batchnorm/Rsqrt:y:0Hsequential_9/batch_normalization_43/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_43/batchnorm/mul?
3sequential_9/batch_normalization_43/batchnorm/mul_1Mul)sequential_9/conv3d_18/Relu:activations:05sequential_9/batch_normalization_43/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 25
3sequential_9/batch_normalization_43/batchnorm/mul_1?
>sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_9_batch_normalization_43_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02@
>sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_1?
3sequential_9/batch_normalization_43/batchnorm/mul_2MulFsequential_9/batch_normalization_43/batchnorm/ReadVariableOp_1:value:05sequential_9/batch_normalization_43/batchnorm/mul:z:0*
T0*
_output_shapes
: 25
3sequential_9/batch_normalization_43/batchnorm/mul_2?
>sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_9_batch_normalization_43_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02@
>sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_2?
1sequential_9/batch_normalization_43/batchnorm/subSubFsequential_9/batch_normalization_43/batchnorm/ReadVariableOp_2:value:07sequential_9/batch_normalization_43/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_43/batchnorm/sub?
3sequential_9/batch_normalization_43/batchnorm/add_1AddV27sequential_9/batch_normalization_43/batchnorm/mul_1:z:05sequential_9/batch_normalization_43/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 25
3sequential_9/batch_normalization_43/batchnorm/add_1?
'sequential_9/max_pooling3d_33/MaxPool3D	MaxPool3D7sequential_9/batch_normalization_43/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2)
'sequential_9/max_pooling3d_33/MaxPool3D?
,sequential_9/conv3d_19/Conv3D/ReadVariableOpReadVariableOp5sequential_9_conv3d_19_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02.
,sequential_9/conv3d_19/Conv3D/ReadVariableOp?
sequential_9/conv3d_19/Conv3DConv3D0sequential_9/max_pooling3d_33/MaxPool3D:output:04sequential_9/conv3d_19/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
sequential_9/conv3d_19/Conv3D?
-sequential_9/conv3d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv3d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv3d_19/BiasAdd/ReadVariableOp?
sequential_9/conv3d_19/BiasAddBiasAdd&sequential_9/conv3d_19/Conv3D:output:05sequential_9/conv3d_19/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2 
sequential_9/conv3d_19/BiasAdd?
sequential_9/conv3d_19/ReluRelu'sequential_9/conv3d_19/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
sequential_9/conv3d_19/Relu?
<sequential_9/batch_normalization_44/batchnorm/ReadVariableOpReadVariableOpEsequential_9_batch_normalization_44_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02>
<sequential_9/batch_normalization_44/batchnorm/ReadVariableOp?
3sequential_9/batch_normalization_44/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_9/batch_normalization_44/batchnorm/add/y?
1sequential_9/batch_normalization_44/batchnorm/addAddV2Dsequential_9/batch_normalization_44/batchnorm/ReadVariableOp:value:0<sequential_9/batch_normalization_44/batchnorm/add/y:output:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_44/batchnorm/add?
3sequential_9/batch_normalization_44/batchnorm/RsqrtRsqrt5sequential_9/batch_normalization_44/batchnorm/add:z:0*
T0*
_output_shapes
: 25
3sequential_9/batch_normalization_44/batchnorm/Rsqrt?
@sequential_9/batch_normalization_44/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_9_batch_normalization_44_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02B
@sequential_9/batch_normalization_44/batchnorm/mul/ReadVariableOp?
1sequential_9/batch_normalization_44/batchnorm/mulMul7sequential_9/batch_normalization_44/batchnorm/Rsqrt:y:0Hsequential_9/batch_normalization_44/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_44/batchnorm/mul?
3sequential_9/batch_normalization_44/batchnorm/mul_1Mul)sequential_9/conv3d_19/Relu:activations:05sequential_9/batch_normalization_44/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 25
3sequential_9/batch_normalization_44/batchnorm/mul_1?
>sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_9_batch_normalization_44_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02@
>sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_1?
3sequential_9/batch_normalization_44/batchnorm/mul_2MulFsequential_9/batch_normalization_44/batchnorm/ReadVariableOp_1:value:05sequential_9/batch_normalization_44/batchnorm/mul:z:0*
T0*
_output_shapes
: 25
3sequential_9/batch_normalization_44/batchnorm/mul_2?
>sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_9_batch_normalization_44_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02@
>sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_2?
1sequential_9/batch_normalization_44/batchnorm/subSubFsequential_9/batch_normalization_44/batchnorm/ReadVariableOp_2:value:07sequential_9/batch_normalization_44/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 23
1sequential_9/batch_normalization_44/batchnorm/sub?
3sequential_9/batch_normalization_44/batchnorm/add_1AddV27sequential_9/batch_normalization_44/batchnorm/mul_1:z:05sequential_9/batch_normalization_44/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 25
3sequential_9/batch_normalization_44/batchnorm/add_1?
'sequential_9/max_pooling3d_34/MaxPool3D	MaxPool3D7sequential_9/batch_normalization_44/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2)
'sequential_9/max_pooling3d_34/MaxPool3D?
sequential_9/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_9/flatten_8/Const?
sequential_9/flatten_8/ReshapeReshape0sequential_9/max_pooling3d_34/MaxPool3D:output:0%sequential_9/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????
2 
sequential_9/flatten_8/Reshape?
 sequential_9/dropout_16/IdentityIdentity'sequential_9/flatten_8/Reshape:output:0*
T0*(
_output_shapes
:??????????
2"
 sequential_9/dropout_16/Identity?
+sequential_9/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02-
+sequential_9/dense_16/MatMul/ReadVariableOp?
sequential_9/dense_16/MatMulMatMul)sequential_9/dropout_16/Identity:output:03sequential_9/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_16/MatMul?
,sequential_9/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_9/dense_16/BiasAdd/ReadVariableOp?
sequential_9/dense_16/BiasAddBiasAdd&sequential_9/dense_16/MatMul:product:04sequential_9/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_16/BiasAdd?
sequential_9/dense_16/ReluRelu&sequential_9/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_16/Relu?
 sequential_9/dropout_17/IdentityIdentity(sequential_9/dense_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 sequential_9/dropout_17/Identity?
+sequential_9/dense_17/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+sequential_9/dense_17/MatMul/ReadVariableOp?
sequential_9/dense_17/MatMulMatMul)sequential_9/dropout_17/Identity:output:03sequential_9/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_17/MatMul?
,sequential_9/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_9/dense_17/BiasAdd/ReadVariableOp?
sequential_9/dense_17/BiasAddBiasAdd&sequential_9/dense_17/MatMul:product:04sequential_9/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_17/BiasAdd?
sequential_9/dense_17/SoftmaxSoftmax&sequential_9/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_17/Softmax?
IdentityIdentity'sequential_9/dense_17/Softmax:softmax:0=^sequential_9/batch_normalization_40/batchnorm/ReadVariableOp?^sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_1?^sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_2A^sequential_9/batch_normalization_40/batchnorm/mul/ReadVariableOp=^sequential_9/batch_normalization_41/batchnorm/ReadVariableOp?^sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_1?^sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_2A^sequential_9/batch_normalization_41/batchnorm/mul/ReadVariableOp=^sequential_9/batch_normalization_42/batchnorm/ReadVariableOp?^sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_1?^sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_2A^sequential_9/batch_normalization_42/batchnorm/mul/ReadVariableOp=^sequential_9/batch_normalization_43/batchnorm/ReadVariableOp?^sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_1?^sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_2A^sequential_9/batch_normalization_43/batchnorm/mul/ReadVariableOp=^sequential_9/batch_normalization_44/batchnorm/ReadVariableOp?^sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_1?^sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_2A^sequential_9/batch_normalization_44/batchnorm/mul/ReadVariableOp.^sequential_9/conv3d_15/BiasAdd/ReadVariableOp-^sequential_9/conv3d_15/Conv3D/ReadVariableOp.^sequential_9/conv3d_16/BiasAdd/ReadVariableOp-^sequential_9/conv3d_16/Conv3D/ReadVariableOp.^sequential_9/conv3d_17/BiasAdd/ReadVariableOp-^sequential_9/conv3d_17/Conv3D/ReadVariableOp.^sequential_9/conv3d_18/BiasAdd/ReadVariableOp-^sequential_9/conv3d_18/Conv3D/ReadVariableOp.^sequential_9/conv3d_19/BiasAdd/ReadVariableOp-^sequential_9/conv3d_19/Conv3D/ReadVariableOp-^sequential_9/dense_16/BiasAdd/ReadVariableOp,^sequential_9/dense_16/MatMul/ReadVariableOp-^sequential_9/dense_17/BiasAdd/ReadVariableOp,^sequential_9/dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2|
<sequential_9/batch_normalization_40/batchnorm/ReadVariableOp<sequential_9/batch_normalization_40/batchnorm/ReadVariableOp2?
>sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_1>sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_12?
>sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_2>sequential_9/batch_normalization_40/batchnorm/ReadVariableOp_22?
@sequential_9/batch_normalization_40/batchnorm/mul/ReadVariableOp@sequential_9/batch_normalization_40/batchnorm/mul/ReadVariableOp2|
<sequential_9/batch_normalization_41/batchnorm/ReadVariableOp<sequential_9/batch_normalization_41/batchnorm/ReadVariableOp2?
>sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_1>sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_12?
>sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_2>sequential_9/batch_normalization_41/batchnorm/ReadVariableOp_22?
@sequential_9/batch_normalization_41/batchnorm/mul/ReadVariableOp@sequential_9/batch_normalization_41/batchnorm/mul/ReadVariableOp2|
<sequential_9/batch_normalization_42/batchnorm/ReadVariableOp<sequential_9/batch_normalization_42/batchnorm/ReadVariableOp2?
>sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_1>sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_12?
>sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_2>sequential_9/batch_normalization_42/batchnorm/ReadVariableOp_22?
@sequential_9/batch_normalization_42/batchnorm/mul/ReadVariableOp@sequential_9/batch_normalization_42/batchnorm/mul/ReadVariableOp2|
<sequential_9/batch_normalization_43/batchnorm/ReadVariableOp<sequential_9/batch_normalization_43/batchnorm/ReadVariableOp2?
>sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_1>sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_12?
>sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_2>sequential_9/batch_normalization_43/batchnorm/ReadVariableOp_22?
@sequential_9/batch_normalization_43/batchnorm/mul/ReadVariableOp@sequential_9/batch_normalization_43/batchnorm/mul/ReadVariableOp2|
<sequential_9/batch_normalization_44/batchnorm/ReadVariableOp<sequential_9/batch_normalization_44/batchnorm/ReadVariableOp2?
>sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_1>sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_12?
>sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_2>sequential_9/batch_normalization_44/batchnorm/ReadVariableOp_22?
@sequential_9/batch_normalization_44/batchnorm/mul/ReadVariableOp@sequential_9/batch_normalization_44/batchnorm/mul/ReadVariableOp2^
-sequential_9/conv3d_15/BiasAdd/ReadVariableOp-sequential_9/conv3d_15/BiasAdd/ReadVariableOp2\
,sequential_9/conv3d_15/Conv3D/ReadVariableOp,sequential_9/conv3d_15/Conv3D/ReadVariableOp2^
-sequential_9/conv3d_16/BiasAdd/ReadVariableOp-sequential_9/conv3d_16/BiasAdd/ReadVariableOp2\
,sequential_9/conv3d_16/Conv3D/ReadVariableOp,sequential_9/conv3d_16/Conv3D/ReadVariableOp2^
-sequential_9/conv3d_17/BiasAdd/ReadVariableOp-sequential_9/conv3d_17/BiasAdd/ReadVariableOp2\
,sequential_9/conv3d_17/Conv3D/ReadVariableOp,sequential_9/conv3d_17/Conv3D/ReadVariableOp2^
-sequential_9/conv3d_18/BiasAdd/ReadVariableOp-sequential_9/conv3d_18/BiasAdd/ReadVariableOp2\
,sequential_9/conv3d_18/Conv3D/ReadVariableOp,sequential_9/conv3d_18/Conv3D/ReadVariableOp2^
-sequential_9/conv3d_19/BiasAdd/ReadVariableOp-sequential_9/conv3d_19/BiasAdd/ReadVariableOp2\
,sequential_9/conv3d_19/Conv3D/ReadVariableOp,sequential_9/conv3d_19/Conv3D/ReadVariableOp2\
,sequential_9/dense_16/BiasAdd/ReadVariableOp,sequential_9/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_16/MatMul/ReadVariableOp+sequential_9/dense_16/MatMul/ReadVariableOp2\
,sequential_9/dense_17/BiasAdd/ReadVariableOp,sequential_9/dense_17/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_17/MatMul/ReadVariableOp+sequential_9/dense_17/MatMul/ReadVariableOp:d `
3
_output_shapes!
:?????????
@@
)
_user_specified_nameconv3d_15_input
Β
? 
H__inference_sequential_9_layer_call_and_return_conditional_losses_420827

inputsF
(conv3d_15_conv3d_readvariableop_resource:
 7
)conv3d_15_biasadd_readvariableop_resource: F
8batch_normalization_40_batchnorm_readvariableop_resource: J
<batch_normalization_40_batchnorm_mul_readvariableop_resource: H
:batch_normalization_40_batchnorm_readvariableop_1_resource: H
:batch_normalization_40_batchnorm_readvariableop_2_resource: F
(conv3d_16_conv3d_readvariableop_resource:
  7
)conv3d_16_biasadd_readvariableop_resource: F
8batch_normalization_41_batchnorm_readvariableop_resource: J
<batch_normalization_41_batchnorm_mul_readvariableop_resource: H
:batch_normalization_41_batchnorm_readvariableop_1_resource: H
:batch_normalization_41_batchnorm_readvariableop_2_resource: F
(conv3d_17_conv3d_readvariableop_resource:
  7
)conv3d_17_biasadd_readvariableop_resource: F
8batch_normalization_42_batchnorm_readvariableop_resource: J
<batch_normalization_42_batchnorm_mul_readvariableop_resource: H
:batch_normalization_42_batchnorm_readvariableop_1_resource: H
:batch_normalization_42_batchnorm_readvariableop_2_resource: F
(conv3d_18_conv3d_readvariableop_resource:
  7
)conv3d_18_biasadd_readvariableop_resource: F
8batch_normalization_43_batchnorm_readvariableop_resource: J
<batch_normalization_43_batchnorm_mul_readvariableop_resource: H
:batch_normalization_43_batchnorm_readvariableop_1_resource: H
:batch_normalization_43_batchnorm_readvariableop_2_resource: F
(conv3d_19_conv3d_readvariableop_resource:
  7
)conv3d_19_biasadd_readvariableop_resource: F
8batch_normalization_44_batchnorm_readvariableop_resource: J
<batch_normalization_44_batchnorm_mul_readvariableop_resource: H
:batch_normalization_44_batchnorm_readvariableop_1_resource: H
:batch_normalization_44_batchnorm_readvariableop_2_resource: ;
'dense_16_matmul_readvariableop_resource:
?
?7
(dense_16_biasadd_readvariableop_resource:	?:
'dense_17_matmul_readvariableop_resource:	?6
(dense_17_biasadd_readvariableop_resource:
identity??/batch_normalization_40/batchnorm/ReadVariableOp?1batch_normalization_40/batchnorm/ReadVariableOp_1?1batch_normalization_40/batchnorm/ReadVariableOp_2?3batch_normalization_40/batchnorm/mul/ReadVariableOp?/batch_normalization_41/batchnorm/ReadVariableOp?1batch_normalization_41/batchnorm/ReadVariableOp_1?1batch_normalization_41/batchnorm/ReadVariableOp_2?3batch_normalization_41/batchnorm/mul/ReadVariableOp?/batch_normalization_42/batchnorm/ReadVariableOp?1batch_normalization_42/batchnorm/ReadVariableOp_1?1batch_normalization_42/batchnorm/ReadVariableOp_2?3batch_normalization_42/batchnorm/mul/ReadVariableOp?/batch_normalization_43/batchnorm/ReadVariableOp?1batch_normalization_43/batchnorm/ReadVariableOp_1?1batch_normalization_43/batchnorm/ReadVariableOp_2?3batch_normalization_43/batchnorm/mul/ReadVariableOp?/batch_normalization_44/batchnorm/ReadVariableOp?1batch_normalization_44/batchnorm/ReadVariableOp_1?1batch_normalization_44/batchnorm/ReadVariableOp_2?3batch_normalization_44/batchnorm/mul/ReadVariableOp? conv3d_15/BiasAdd/ReadVariableOp?conv3d_15/Conv3D/ReadVariableOp? conv3d_16/BiasAdd/ReadVariableOp?conv3d_16/Conv3D/ReadVariableOp?2conv3d_16/kernel/Regularizer/Square/ReadVariableOp? conv3d_17/BiasAdd/ReadVariableOp?conv3d_17/Conv3D/ReadVariableOp?2conv3d_17/kernel/Regularizer/Square/ReadVariableOp? conv3d_18/BiasAdd/ReadVariableOp?conv3d_18/Conv3D/ReadVariableOp?2conv3d_18/kernel/Regularizer/Square/ReadVariableOp? conv3d_19/BiasAdd/ReadVariableOp?conv3d_19/Conv3D/ReadVariableOp?2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?
conv3d_15/Conv3D/ReadVariableOpReadVariableOp(conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:
 *
dtype02!
conv3d_15/Conv3D/ReadVariableOp?
conv3d_15/Conv3DConv3Dinputs'conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
@@ *
paddingSAME*
strides	
2
conv3d_15/Conv3D?
 conv3d_15/BiasAdd/ReadVariableOpReadVariableOp)conv3d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_15/BiasAdd/ReadVariableOp?
conv3d_15/BiasAddBiasAddconv3d_15/Conv3D:output:0(conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
@@ 2
conv3d_15/BiasAdd?
conv3d_15/ReluReluconv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
@@ 2
conv3d_15/Relu?
/batch_normalization_40/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_40_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_40/batchnorm/ReadVariableOp?
&batch_normalization_40/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_40/batchnorm/add/y?
$batch_normalization_40/batchnorm/addAddV27batch_normalization_40/batchnorm/ReadVariableOp:value:0/batch_normalization_40/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_40/batchnorm/add?
&batch_normalization_40/batchnorm/RsqrtRsqrt(batch_normalization_40/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_40/batchnorm/Rsqrt?
3batch_normalization_40/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_40_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_40/batchnorm/mul/ReadVariableOp?
$batch_normalization_40/batchnorm/mulMul*batch_normalization_40/batchnorm/Rsqrt:y:0;batch_normalization_40/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_40/batchnorm/mul?
&batch_normalization_40/batchnorm/mul_1Mulconv3d_15/Relu:activations:0(batch_normalization_40/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
@@ 2(
&batch_normalization_40/batchnorm/mul_1?
1batch_normalization_40/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_40_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_40/batchnorm/ReadVariableOp_1?
&batch_normalization_40/batchnorm/mul_2Mul9batch_normalization_40/batchnorm/ReadVariableOp_1:value:0(batch_normalization_40/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_40/batchnorm/mul_2?
1batch_normalization_40/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_40_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_40/batchnorm/ReadVariableOp_2?
$batch_normalization_40/batchnorm/subSub9batch_normalization_40/batchnorm/ReadVariableOp_2:value:0*batch_normalization_40/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_40/batchnorm/sub?
&batch_normalization_40/batchnorm/add_1AddV2*batch_normalization_40/batchnorm/mul_1:z:0(batch_normalization_40/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
@@ 2(
&batch_normalization_40/batchnorm/add_1?
max_pooling3d_30/MaxPool3D	MaxPool3D*batch_normalization_40/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
   *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_30/MaxPool3D?
conv3d_16/Conv3D/ReadVariableOpReadVariableOp(conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02!
conv3d_16/Conv3D/ReadVariableOp?
conv3d_16/Conv3DConv3D#max_pooling3d_30/MaxPool3D:output:0'conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
   *
paddingSAME*
strides	
2
conv3d_16/Conv3D?
 conv3d_16/BiasAdd/ReadVariableOpReadVariableOp)conv3d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_16/BiasAdd/ReadVariableOp?
conv3d_16/BiasAddBiasAddconv3d_16/Conv3D:output:0(conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
   2
conv3d_16/BiasAdd?
conv3d_16/ReluReluconv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
   2
conv3d_16/Relu?
/batch_normalization_41/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_41_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_41/batchnorm/ReadVariableOp?
&batch_normalization_41/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_41/batchnorm/add/y?
$batch_normalization_41/batchnorm/addAddV27batch_normalization_41/batchnorm/ReadVariableOp:value:0/batch_normalization_41/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_41/batchnorm/add?
&batch_normalization_41/batchnorm/RsqrtRsqrt(batch_normalization_41/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_41/batchnorm/Rsqrt?
3batch_normalization_41/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_41_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_41/batchnorm/mul/ReadVariableOp?
$batch_normalization_41/batchnorm/mulMul*batch_normalization_41/batchnorm/Rsqrt:y:0;batch_normalization_41/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_41/batchnorm/mul?
&batch_normalization_41/batchnorm/mul_1Mulconv3d_16/Relu:activations:0(batch_normalization_41/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
   2(
&batch_normalization_41/batchnorm/mul_1?
1batch_normalization_41/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_41_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_41/batchnorm/ReadVariableOp_1?
&batch_normalization_41/batchnorm/mul_2Mul9batch_normalization_41/batchnorm/ReadVariableOp_1:value:0(batch_normalization_41/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_41/batchnorm/mul_2?
1batch_normalization_41/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_41_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_41/batchnorm/ReadVariableOp_2?
$batch_normalization_41/batchnorm/subSub9batch_normalization_41/batchnorm/ReadVariableOp_2:value:0*batch_normalization_41/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_41/batchnorm/sub?
&batch_normalization_41/batchnorm/add_1AddV2*batch_normalization_41/batchnorm/mul_1:z:0(batch_normalization_41/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
   2(
&batch_normalization_41/batchnorm/add_1?
max_pooling3d_31/MaxPool3D	MaxPool3D*batch_normalization_41/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_31/MaxPool3D?
conv3d_17/Conv3D/ReadVariableOpReadVariableOp(conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02!
conv3d_17/Conv3D/ReadVariableOp?
conv3d_17/Conv3DConv3D#max_pooling3d_31/MaxPool3D:output:0'conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
conv3d_17/Conv3D?
 conv3d_17/BiasAdd/ReadVariableOpReadVariableOp)conv3d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_17/BiasAdd/ReadVariableOp?
conv3d_17/BiasAddBiasAddconv3d_17/Conv3D:output:0(conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_17/BiasAdd?
conv3d_17/ReluReluconv3d_17/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_17/Relu?
/batch_normalization_42/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_42_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_42/batchnorm/ReadVariableOp?
&batch_normalization_42/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_42/batchnorm/add/y?
$batch_normalization_42/batchnorm/addAddV27batch_normalization_42/batchnorm/ReadVariableOp:value:0/batch_normalization_42/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_42/batchnorm/add?
&batch_normalization_42/batchnorm/RsqrtRsqrt(batch_normalization_42/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_42/batchnorm/Rsqrt?
3batch_normalization_42/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_42_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_42/batchnorm/mul/ReadVariableOp?
$batch_normalization_42/batchnorm/mulMul*batch_normalization_42/batchnorm/Rsqrt:y:0;batch_normalization_42/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_42/batchnorm/mul?
&batch_normalization_42/batchnorm/mul_1Mulconv3d_17/Relu:activations:0(batch_normalization_42/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_42/batchnorm/mul_1?
1batch_normalization_42/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_42_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_42/batchnorm/ReadVariableOp_1?
&batch_normalization_42/batchnorm/mul_2Mul9batch_normalization_42/batchnorm/ReadVariableOp_1:value:0(batch_normalization_42/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_42/batchnorm/mul_2?
1batch_normalization_42/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_42_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_42/batchnorm/ReadVariableOp_2?
$batch_normalization_42/batchnorm/subSub9batch_normalization_42/batchnorm/ReadVariableOp_2:value:0*batch_normalization_42/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_42/batchnorm/sub?
&batch_normalization_42/batchnorm/add_1AddV2*batch_normalization_42/batchnorm/mul_1:z:0(batch_normalization_42/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_42/batchnorm/add_1?
max_pooling3d_32/MaxPool3D	MaxPool3D*batch_normalization_42/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_32/MaxPool3D?
conv3d_18/Conv3D/ReadVariableOpReadVariableOp(conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02!
conv3d_18/Conv3D/ReadVariableOp?
conv3d_18/Conv3DConv3D#max_pooling3d_32/MaxPool3D:output:0'conv3d_18/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
conv3d_18/Conv3D?
 conv3d_18/BiasAdd/ReadVariableOpReadVariableOp)conv3d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_18/BiasAdd/ReadVariableOp?
conv3d_18/BiasAddBiasAddconv3d_18/Conv3D:output:0(conv3d_18/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_18/BiasAdd?
conv3d_18/ReluReluconv3d_18/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_18/Relu?
/batch_normalization_43/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_43_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_43/batchnorm/ReadVariableOp?
&batch_normalization_43/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_43/batchnorm/add/y?
$batch_normalization_43/batchnorm/addAddV27batch_normalization_43/batchnorm/ReadVariableOp:value:0/batch_normalization_43/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_43/batchnorm/add?
&batch_normalization_43/batchnorm/RsqrtRsqrt(batch_normalization_43/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_43/batchnorm/Rsqrt?
3batch_normalization_43/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_43_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_43/batchnorm/mul/ReadVariableOp?
$batch_normalization_43/batchnorm/mulMul*batch_normalization_43/batchnorm/Rsqrt:y:0;batch_normalization_43/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_43/batchnorm/mul?
&batch_normalization_43/batchnorm/mul_1Mulconv3d_18/Relu:activations:0(batch_normalization_43/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_43/batchnorm/mul_1?
1batch_normalization_43/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_43_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_43/batchnorm/ReadVariableOp_1?
&batch_normalization_43/batchnorm/mul_2Mul9batch_normalization_43/batchnorm/ReadVariableOp_1:value:0(batch_normalization_43/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_43/batchnorm/mul_2?
1batch_normalization_43/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_43_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_43/batchnorm/ReadVariableOp_2?
$batch_normalization_43/batchnorm/subSub9batch_normalization_43/batchnorm/ReadVariableOp_2:value:0*batch_normalization_43/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_43/batchnorm/sub?
&batch_normalization_43/batchnorm/add_1AddV2*batch_normalization_43/batchnorm/mul_1:z:0(batch_normalization_43/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_43/batchnorm/add_1?
max_pooling3d_33/MaxPool3D	MaxPool3D*batch_normalization_43/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_33/MaxPool3D?
conv3d_19/Conv3D/ReadVariableOpReadVariableOp(conv3d_19_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02!
conv3d_19/Conv3D/ReadVariableOp?
conv3d_19/Conv3DConv3D#max_pooling3d_33/MaxPool3D:output:0'conv3d_19/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
conv3d_19/Conv3D?
 conv3d_19/BiasAdd/ReadVariableOpReadVariableOp)conv3d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_19/BiasAdd/ReadVariableOp?
conv3d_19/BiasAddBiasAddconv3d_19/Conv3D:output:0(conv3d_19/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_19/BiasAdd?
conv3d_19/ReluReluconv3d_19/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_19/Relu?
/batch_normalization_44/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_44_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_44/batchnorm/ReadVariableOp?
&batch_normalization_44/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_44/batchnorm/add/y?
$batch_normalization_44/batchnorm/addAddV27batch_normalization_44/batchnorm/ReadVariableOp:value:0/batch_normalization_44/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_44/batchnorm/add?
&batch_normalization_44/batchnorm/RsqrtRsqrt(batch_normalization_44/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_44/batchnorm/Rsqrt?
3batch_normalization_44/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_44_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_44/batchnorm/mul/ReadVariableOp?
$batch_normalization_44/batchnorm/mulMul*batch_normalization_44/batchnorm/Rsqrt:y:0;batch_normalization_44/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_44/batchnorm/mul?
&batch_normalization_44/batchnorm/mul_1Mulconv3d_19/Relu:activations:0(batch_normalization_44/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_44/batchnorm/mul_1?
1batch_normalization_44/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_44_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_44/batchnorm/ReadVariableOp_1?
&batch_normalization_44/batchnorm/mul_2Mul9batch_normalization_44/batchnorm/ReadVariableOp_1:value:0(batch_normalization_44/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_44/batchnorm/mul_2?
1batch_normalization_44/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_44_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_44/batchnorm/ReadVariableOp_2?
$batch_normalization_44/batchnorm/subSub9batch_normalization_44/batchnorm/ReadVariableOp_2:value:0*batch_normalization_44/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_44/batchnorm/sub?
&batch_normalization_44/batchnorm/add_1AddV2*batch_normalization_44/batchnorm/mul_1:z:0(batch_normalization_44/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_44/batchnorm/add_1?
max_pooling3d_34/MaxPool3D	MaxPool3D*batch_normalization_44/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_34/MaxPool3Ds
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_8/Const?
flatten_8/ReshapeReshape#max_pooling3d_34/MaxPool3D:output:0flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????
2
flatten_8/Reshape?
dropout_16/IdentityIdentityflatten_8/Reshape:output:0*
T0*(
_output_shapes
:??????????
2
dropout_16/Identity?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMuldropout_16/Identity:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/BiasAddt
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_16/Relu?
dropout_17/IdentityIdentitydense_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_17/Identity?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Softmax?
2conv3d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_16/kernel/Regularizer/SquareSquare:conv3d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_16/kernel/Regularizer/Square?
"conv3d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_16/kernel/Regularizer/Const?
 conv3d_16/kernel/Regularizer/SumSum'conv3d_16/kernel/Regularizer/Square:y:0+conv3d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/Sum?
"conv3d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_16/kernel/Regularizer/mul/x?
 conv3d_16/kernel/Regularizer/mulMul+conv3d_16/kernel/Regularizer/mul/x:output:0)conv3d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/mul?
2conv3d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_17/kernel/Regularizer/SquareSquare:conv3d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_17/kernel/Regularizer/Square?
"conv3d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_17/kernel/Regularizer/Const?
 conv3d_17/kernel/Regularizer/SumSum'conv3d_17/kernel/Regularizer/Square:y:0+conv3d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/Sum?
"conv3d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_17/kernel/Regularizer/mul/x?
 conv3d_17/kernel/Regularizer/mulMul+conv3d_17/kernel/Regularizer/mul/x:output:0)conv3d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/mul?
2conv3d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_18/kernel/Regularizer/SquareSquare:conv3d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_18/kernel/Regularizer/Square?
"conv3d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_18/kernel/Regularizer/Const?
 conv3d_18/kernel/Regularizer/SumSum'conv3d_18/kernel/Regularizer/Square:y:0+conv3d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/Sum?
"conv3d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_18/kernel/Regularizer/mul/x?
 conv3d_18/kernel/Regularizer/mulMul+conv3d_18/kernel/Regularizer/mul/x:output:0)conv3d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/mul?
2conv3d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv3d_19_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_19/kernel/Regularizer/SquareSquare:conv3d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_19/kernel/Regularizer/Square?
"conv3d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_19/kernel/Regularizer/Const?
 conv3d_19/kernel/Regularizer/SumSum'conv3d_19/kernel/Regularizer/Square:y:0+conv3d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/Sum?
"conv3d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_19/kernel/Regularizer/mul/x?
 conv3d_19/kernel/Regularizer/mulMul+conv3d_19/kernel/Regularizer/mul/x:output:0)conv3d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/mul?
IdentityIdentitydense_17/Softmax:softmax:00^batch_normalization_40/batchnorm/ReadVariableOp2^batch_normalization_40/batchnorm/ReadVariableOp_12^batch_normalization_40/batchnorm/ReadVariableOp_24^batch_normalization_40/batchnorm/mul/ReadVariableOp0^batch_normalization_41/batchnorm/ReadVariableOp2^batch_normalization_41/batchnorm/ReadVariableOp_12^batch_normalization_41/batchnorm/ReadVariableOp_24^batch_normalization_41/batchnorm/mul/ReadVariableOp0^batch_normalization_42/batchnorm/ReadVariableOp2^batch_normalization_42/batchnorm/ReadVariableOp_12^batch_normalization_42/batchnorm/ReadVariableOp_24^batch_normalization_42/batchnorm/mul/ReadVariableOp0^batch_normalization_43/batchnorm/ReadVariableOp2^batch_normalization_43/batchnorm/ReadVariableOp_12^batch_normalization_43/batchnorm/ReadVariableOp_24^batch_normalization_43/batchnorm/mul/ReadVariableOp0^batch_normalization_44/batchnorm/ReadVariableOp2^batch_normalization_44/batchnorm/ReadVariableOp_12^batch_normalization_44/batchnorm/ReadVariableOp_24^batch_normalization_44/batchnorm/mul/ReadVariableOp!^conv3d_15/BiasAdd/ReadVariableOp ^conv3d_15/Conv3D/ReadVariableOp!^conv3d_16/BiasAdd/ReadVariableOp ^conv3d_16/Conv3D/ReadVariableOp3^conv3d_16/kernel/Regularizer/Square/ReadVariableOp!^conv3d_17/BiasAdd/ReadVariableOp ^conv3d_17/Conv3D/ReadVariableOp3^conv3d_17/kernel/Regularizer/Square/ReadVariableOp!^conv3d_18/BiasAdd/ReadVariableOp ^conv3d_18/Conv3D/ReadVariableOp3^conv3d_18/kernel/Regularizer/Square/ReadVariableOp!^conv3d_19/BiasAdd/ReadVariableOp ^conv3d_19/Conv3D/ReadVariableOp3^conv3d_19/kernel/Regularizer/Square/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_40/batchnorm/ReadVariableOp/batch_normalization_40/batchnorm/ReadVariableOp2f
1batch_normalization_40/batchnorm/ReadVariableOp_11batch_normalization_40/batchnorm/ReadVariableOp_12f
1batch_normalization_40/batchnorm/ReadVariableOp_21batch_normalization_40/batchnorm/ReadVariableOp_22j
3batch_normalization_40/batchnorm/mul/ReadVariableOp3batch_normalization_40/batchnorm/mul/ReadVariableOp2b
/batch_normalization_41/batchnorm/ReadVariableOp/batch_normalization_41/batchnorm/ReadVariableOp2f
1batch_normalization_41/batchnorm/ReadVariableOp_11batch_normalization_41/batchnorm/ReadVariableOp_12f
1batch_normalization_41/batchnorm/ReadVariableOp_21batch_normalization_41/batchnorm/ReadVariableOp_22j
3batch_normalization_41/batchnorm/mul/ReadVariableOp3batch_normalization_41/batchnorm/mul/ReadVariableOp2b
/batch_normalization_42/batchnorm/ReadVariableOp/batch_normalization_42/batchnorm/ReadVariableOp2f
1batch_normalization_42/batchnorm/ReadVariableOp_11batch_normalization_42/batchnorm/ReadVariableOp_12f
1batch_normalization_42/batchnorm/ReadVariableOp_21batch_normalization_42/batchnorm/ReadVariableOp_22j
3batch_normalization_42/batchnorm/mul/ReadVariableOp3batch_normalization_42/batchnorm/mul/ReadVariableOp2b
/batch_normalization_43/batchnorm/ReadVariableOp/batch_normalization_43/batchnorm/ReadVariableOp2f
1batch_normalization_43/batchnorm/ReadVariableOp_11batch_normalization_43/batchnorm/ReadVariableOp_12f
1batch_normalization_43/batchnorm/ReadVariableOp_21batch_normalization_43/batchnorm/ReadVariableOp_22j
3batch_normalization_43/batchnorm/mul/ReadVariableOp3batch_normalization_43/batchnorm/mul/ReadVariableOp2b
/batch_normalization_44/batchnorm/ReadVariableOp/batch_normalization_44/batchnorm/ReadVariableOp2f
1batch_normalization_44/batchnorm/ReadVariableOp_11batch_normalization_44/batchnorm/ReadVariableOp_12f
1batch_normalization_44/batchnorm/ReadVariableOp_21batch_normalization_44/batchnorm/ReadVariableOp_22j
3batch_normalization_44/batchnorm/mul/ReadVariableOp3batch_normalization_44/batchnorm/mul/ReadVariableOp2D
 conv3d_15/BiasAdd/ReadVariableOp conv3d_15/BiasAdd/ReadVariableOp2B
conv3d_15/Conv3D/ReadVariableOpconv3d_15/Conv3D/ReadVariableOp2D
 conv3d_16/BiasAdd/ReadVariableOp conv3d_16/BiasAdd/ReadVariableOp2B
conv3d_16/Conv3D/ReadVariableOpconv3d_16/Conv3D/ReadVariableOp2h
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_17/BiasAdd/ReadVariableOp conv3d_17/BiasAdd/ReadVariableOp2B
conv3d_17/Conv3D/ReadVariableOpconv3d_17/Conv3D/ReadVariableOp2h
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_18/BiasAdd/ReadVariableOp conv3d_18/BiasAdd/ReadVariableOp2B
conv3d_18/Conv3D/ReadVariableOpconv3d_18/Conv3D/ReadVariableOp2h
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_19/BiasAdd/ReadVariableOp conv3d_19/BiasAdd/ReadVariableOp2B
conv3d_19/Conv3D/ReadVariableOpconv3d_19/Conv3D/ReadVariableOp2h
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
@@
 
_user_specified_nameinputs
?
?
)__inference_dense_16_layer_call_fn_422072

inputs
unknown:
?
?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_4192952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
E__inference_conv3d_17_layer_call_and_return_conditional_losses_419127

inputs<
conv3d_readvariableop_resource:
  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
Relu?
2conv3d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_17/kernel/Regularizer/SquareSquare:conv3d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_17/kernel/Regularizer/Square?
"conv3d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_17/kernel/Regularizer/Const?
 conv3d_17/kernel/Regularizer/SumSum'conv3d_17/kernel/Regularizer/Square:y:0+conv3d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/Sum?
"conv3d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_17/kernel/Regularizer/mul/x?
 conv3d_17/kernel/Regularizer/mulMul+conv3d_17/kernel/Regularizer/mul/x:output:0)conv3d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp3^conv3d_17/kernel/Regularizer/Square/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2h
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2conv3d_17/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_419152

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?+
?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421641

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
G
+__inference_dropout_17_layer_call_fn_422088

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_4193062
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421169

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv3d_17_layer_call_fn_421464

inputs%
unknown:
  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_4191272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
 : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_40_layer_call_fn_421136

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4190462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
@@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
@@ 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_41_layer_call_fn_421328

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_4190992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
   : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
   
 
_user_specified_nameinputs
?
?
*__inference_conv3d_19_layer_call_fn_421848

inputs%
unknown:
  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_19_layer_call_and_return_conditional_losses_4192332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
 : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_32_layer_call_fn_418655

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_32_layer_call_and_return_conditional_losses_4186492
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_422036

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
 :[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?+
?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_419822

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
@@ 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
@@ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
@@ 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
@@ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
@@ 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_422174Y
;conv3d_19_kernel_regularizer_square_readvariableop_resource:
  
identity??2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
2conv3d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv3d_19_kernel_regularizer_square_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_19/kernel/Regularizer/SquareSquare:conv3d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_19/kernel/Regularizer/Square?
"conv3d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_19/kernel/Regularizer/Const?
 conv3d_19/kernel/Regularizer/SumSum'conv3d_19/kernel/Regularizer/Square:y:0+conv3d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/Sum?
"conv3d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_19/kernel/Regularizer/mul/x?
 conv3d_19/kernel/Regularizer/mulMul+conv3d_19/kernel/Regularizer/mul/x:output:0)conv3d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/mul?
IdentityIdentity$conv3d_19/kernel/Regularizer/mul:z:03^conv3d_19/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2conv3d_19/kernel/Regularizer/Square/ReadVariableOp
?
?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421223

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
@@ 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
@@ 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
@@ : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
@@ 
 
_user_specified_nameinputs
?
?
E__inference_conv3d_15_layer_call_and_return_conditional_losses_419021

inputs<
conv3d_readvariableop_resource:
 -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
 *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
@@ *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
@@ 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
@@ 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????
@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????
@@
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_31_layer_call_fn_418481

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_31_layer_call_and_return_conditional_losses_4184752
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421553

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_419046

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
@@ 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
@@ 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
@@ : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
@@ 
 
_user_specified_nameinputs
?
?
E__inference_conv3d_19_layer_call_and_return_conditional_losses_419233

inputs<
conv3d_readvariableop_resource:
  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
Relu?
2conv3d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_19/kernel/Regularizer/SquareSquare:conv3d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_19/kernel/Regularizer/Square?
"conv3d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_19/kernel/Regularizer/Const?
 conv3d_19/kernel/Regularizer/SumSum'conv3d_19/kernel/Regularizer/Square:y:0+conv3d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/Sum?
"conv3d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_19/kernel/Regularizer/mul/x?
 conv3d_19/kernel/Regularizer/mulMul+conv3d_19/kernel/Regularizer/mul/x:output:0)conv3d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp3^conv3d_19/kernel/Regularizer/Square/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2h
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2conv3d_19/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_44_layer_call_fn_421878

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_4188532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_43_layer_call_fn_421699

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_4187392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_43_layer_call_fn_421686

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_4186792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_419421
conv3d_15_input%
unknown:
 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: '
	unknown_5:
  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: (

unknown_11:
  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: (

unknown_17:
  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: (

unknown_23:
  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29:
?
?

unknown_30:	?

unknown_31:	?

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_4193502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
3
_output_shapes!
:?????????
@@
)
_user_specified_nameconv3d_15_input
??
?$
H__inference_sequential_9_layer_call_and_return_conditional_losses_421077

inputsF
(conv3d_15_conv3d_readvariableop_resource:
 7
)conv3d_15_biasadd_readvariableop_resource: L
>batch_normalization_40_assignmovingavg_readvariableop_resource: N
@batch_normalization_40_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_40_batchnorm_mul_readvariableop_resource: F
8batch_normalization_40_batchnorm_readvariableop_resource: F
(conv3d_16_conv3d_readvariableop_resource:
  7
)conv3d_16_biasadd_readvariableop_resource: L
>batch_normalization_41_assignmovingavg_readvariableop_resource: N
@batch_normalization_41_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_41_batchnorm_mul_readvariableop_resource: F
8batch_normalization_41_batchnorm_readvariableop_resource: F
(conv3d_17_conv3d_readvariableop_resource:
  7
)conv3d_17_biasadd_readvariableop_resource: L
>batch_normalization_42_assignmovingavg_readvariableop_resource: N
@batch_normalization_42_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_42_batchnorm_mul_readvariableop_resource: F
8batch_normalization_42_batchnorm_readvariableop_resource: F
(conv3d_18_conv3d_readvariableop_resource:
  7
)conv3d_18_biasadd_readvariableop_resource: L
>batch_normalization_43_assignmovingavg_readvariableop_resource: N
@batch_normalization_43_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_43_batchnorm_mul_readvariableop_resource: F
8batch_normalization_43_batchnorm_readvariableop_resource: F
(conv3d_19_conv3d_readvariableop_resource:
  7
)conv3d_19_biasadd_readvariableop_resource: L
>batch_normalization_44_assignmovingavg_readvariableop_resource: N
@batch_normalization_44_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_44_batchnorm_mul_readvariableop_resource: F
8batch_normalization_44_batchnorm_readvariableop_resource: ;
'dense_16_matmul_readvariableop_resource:
?
?7
(dense_16_biasadd_readvariableop_resource:	?:
'dense_17_matmul_readvariableop_resource:	?6
(dense_17_biasadd_readvariableop_resource:
identity??&batch_normalization_40/AssignMovingAvg?5batch_normalization_40/AssignMovingAvg/ReadVariableOp?(batch_normalization_40/AssignMovingAvg_1?7batch_normalization_40/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_40/batchnorm/ReadVariableOp?3batch_normalization_40/batchnorm/mul/ReadVariableOp?&batch_normalization_41/AssignMovingAvg?5batch_normalization_41/AssignMovingAvg/ReadVariableOp?(batch_normalization_41/AssignMovingAvg_1?7batch_normalization_41/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_41/batchnorm/ReadVariableOp?3batch_normalization_41/batchnorm/mul/ReadVariableOp?&batch_normalization_42/AssignMovingAvg?5batch_normalization_42/AssignMovingAvg/ReadVariableOp?(batch_normalization_42/AssignMovingAvg_1?7batch_normalization_42/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_42/batchnorm/ReadVariableOp?3batch_normalization_42/batchnorm/mul/ReadVariableOp?&batch_normalization_43/AssignMovingAvg?5batch_normalization_43/AssignMovingAvg/ReadVariableOp?(batch_normalization_43/AssignMovingAvg_1?7batch_normalization_43/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_43/batchnorm/ReadVariableOp?3batch_normalization_43/batchnorm/mul/ReadVariableOp?&batch_normalization_44/AssignMovingAvg?5batch_normalization_44/AssignMovingAvg/ReadVariableOp?(batch_normalization_44/AssignMovingAvg_1?7batch_normalization_44/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_44/batchnorm/ReadVariableOp?3batch_normalization_44/batchnorm/mul/ReadVariableOp? conv3d_15/BiasAdd/ReadVariableOp?conv3d_15/Conv3D/ReadVariableOp? conv3d_16/BiasAdd/ReadVariableOp?conv3d_16/Conv3D/ReadVariableOp?2conv3d_16/kernel/Regularizer/Square/ReadVariableOp? conv3d_17/BiasAdd/ReadVariableOp?conv3d_17/Conv3D/ReadVariableOp?2conv3d_17/kernel/Regularizer/Square/ReadVariableOp? conv3d_18/BiasAdd/ReadVariableOp?conv3d_18/Conv3D/ReadVariableOp?2conv3d_18/kernel/Regularizer/Square/ReadVariableOp? conv3d_19/BiasAdd/ReadVariableOp?conv3d_19/Conv3D/ReadVariableOp?2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?
conv3d_15/Conv3D/ReadVariableOpReadVariableOp(conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:
 *
dtype02!
conv3d_15/Conv3D/ReadVariableOp?
conv3d_15/Conv3DConv3Dinputs'conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
@@ *
paddingSAME*
strides	
2
conv3d_15/Conv3D?
 conv3d_15/BiasAdd/ReadVariableOpReadVariableOp)conv3d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_15/BiasAdd/ReadVariableOp?
conv3d_15/BiasAddBiasAddconv3d_15/Conv3D:output:0(conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
@@ 2
conv3d_15/BiasAdd?
conv3d_15/ReluReluconv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
@@ 2
conv3d_15/Relu?
5batch_normalization_40/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_40/moments/mean/reduction_indices?
#batch_normalization_40/moments/meanMeanconv3d_15/Relu:activations:0>batch_normalization_40/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2%
#batch_normalization_40/moments/mean?
+batch_normalization_40/moments/StopGradientStopGradient,batch_normalization_40/moments/mean:output:0*
T0**
_output_shapes
: 2-
+batch_normalization_40/moments/StopGradient?
0batch_normalization_40/moments/SquaredDifferenceSquaredDifferenceconv3d_15/Relu:activations:04batch_normalization_40/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
@@ 22
0batch_normalization_40/moments/SquaredDifference?
9batch_normalization_40/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_40/moments/variance/reduction_indices?
'batch_normalization_40/moments/varianceMean4batch_normalization_40/moments/SquaredDifference:z:0Bbatch_normalization_40/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2)
'batch_normalization_40/moments/variance?
&batch_normalization_40/moments/SqueezeSqueeze,batch_normalization_40/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_40/moments/Squeeze?
(batch_normalization_40/moments/Squeeze_1Squeeze0batch_normalization_40/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_40/moments/Squeeze_1?
,batch_normalization_40/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_40/AssignMovingAvg/decay?
5batch_normalization_40/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_40_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_40/AssignMovingAvg/ReadVariableOp?
*batch_normalization_40/AssignMovingAvg/subSub=batch_normalization_40/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_40/moments/Squeeze:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_40/AssignMovingAvg/sub?
*batch_normalization_40/AssignMovingAvg/mulMul.batch_normalization_40/AssignMovingAvg/sub:z:05batch_normalization_40/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_40/AssignMovingAvg/mul?
&batch_normalization_40/AssignMovingAvgAssignSubVariableOp>batch_normalization_40_assignmovingavg_readvariableop_resource.batch_normalization_40/AssignMovingAvg/mul:z:06^batch_normalization_40/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_40/AssignMovingAvg?
.batch_normalization_40/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_40/AssignMovingAvg_1/decay?
7batch_normalization_40/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_40_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_40/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_40/AssignMovingAvg_1/subSub?batch_normalization_40/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_40/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_40/AssignMovingAvg_1/sub?
,batch_normalization_40/AssignMovingAvg_1/mulMul0batch_normalization_40/AssignMovingAvg_1/sub:z:07batch_normalization_40/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_40/AssignMovingAvg_1/mul?
(batch_normalization_40/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_40_assignmovingavg_1_readvariableop_resource0batch_normalization_40/AssignMovingAvg_1/mul:z:08^batch_normalization_40/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_40/AssignMovingAvg_1?
&batch_normalization_40/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_40/batchnorm/add/y?
$batch_normalization_40/batchnorm/addAddV21batch_normalization_40/moments/Squeeze_1:output:0/batch_normalization_40/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_40/batchnorm/add?
&batch_normalization_40/batchnorm/RsqrtRsqrt(batch_normalization_40/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_40/batchnorm/Rsqrt?
3batch_normalization_40/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_40_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_40/batchnorm/mul/ReadVariableOp?
$batch_normalization_40/batchnorm/mulMul*batch_normalization_40/batchnorm/Rsqrt:y:0;batch_normalization_40/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_40/batchnorm/mul?
&batch_normalization_40/batchnorm/mul_1Mulconv3d_15/Relu:activations:0(batch_normalization_40/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
@@ 2(
&batch_normalization_40/batchnorm/mul_1?
&batch_normalization_40/batchnorm/mul_2Mul/batch_normalization_40/moments/Squeeze:output:0(batch_normalization_40/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_40/batchnorm/mul_2?
/batch_normalization_40/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_40_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_40/batchnorm/ReadVariableOp?
$batch_normalization_40/batchnorm/subSub7batch_normalization_40/batchnorm/ReadVariableOp:value:0*batch_normalization_40/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_40/batchnorm/sub?
&batch_normalization_40/batchnorm/add_1AddV2*batch_normalization_40/batchnorm/mul_1:z:0(batch_normalization_40/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
@@ 2(
&batch_normalization_40/batchnorm/add_1?
max_pooling3d_30/MaxPool3D	MaxPool3D*batch_normalization_40/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
   *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_30/MaxPool3D?
conv3d_16/Conv3D/ReadVariableOpReadVariableOp(conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02!
conv3d_16/Conv3D/ReadVariableOp?
conv3d_16/Conv3DConv3D#max_pooling3d_30/MaxPool3D:output:0'conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
   *
paddingSAME*
strides	
2
conv3d_16/Conv3D?
 conv3d_16/BiasAdd/ReadVariableOpReadVariableOp)conv3d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_16/BiasAdd/ReadVariableOp?
conv3d_16/BiasAddBiasAddconv3d_16/Conv3D:output:0(conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
   2
conv3d_16/BiasAdd?
conv3d_16/ReluReluconv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
   2
conv3d_16/Relu?
5batch_normalization_41/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_41/moments/mean/reduction_indices?
#batch_normalization_41/moments/meanMeanconv3d_16/Relu:activations:0>batch_normalization_41/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2%
#batch_normalization_41/moments/mean?
+batch_normalization_41/moments/StopGradientStopGradient,batch_normalization_41/moments/mean:output:0*
T0**
_output_shapes
: 2-
+batch_normalization_41/moments/StopGradient?
0batch_normalization_41/moments/SquaredDifferenceSquaredDifferenceconv3d_16/Relu:activations:04batch_normalization_41/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
   22
0batch_normalization_41/moments/SquaredDifference?
9batch_normalization_41/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_41/moments/variance/reduction_indices?
'batch_normalization_41/moments/varianceMean4batch_normalization_41/moments/SquaredDifference:z:0Bbatch_normalization_41/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2)
'batch_normalization_41/moments/variance?
&batch_normalization_41/moments/SqueezeSqueeze,batch_normalization_41/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_41/moments/Squeeze?
(batch_normalization_41/moments/Squeeze_1Squeeze0batch_normalization_41/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_41/moments/Squeeze_1?
,batch_normalization_41/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_41/AssignMovingAvg/decay?
5batch_normalization_41/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_41_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_41/AssignMovingAvg/ReadVariableOp?
*batch_normalization_41/AssignMovingAvg/subSub=batch_normalization_41/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_41/moments/Squeeze:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_41/AssignMovingAvg/sub?
*batch_normalization_41/AssignMovingAvg/mulMul.batch_normalization_41/AssignMovingAvg/sub:z:05batch_normalization_41/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_41/AssignMovingAvg/mul?
&batch_normalization_41/AssignMovingAvgAssignSubVariableOp>batch_normalization_41_assignmovingavg_readvariableop_resource.batch_normalization_41/AssignMovingAvg/mul:z:06^batch_normalization_41/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_41/AssignMovingAvg?
.batch_normalization_41/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_41/AssignMovingAvg_1/decay?
7batch_normalization_41/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_41_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_41/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_41/AssignMovingAvg_1/subSub?batch_normalization_41/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_41/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_41/AssignMovingAvg_1/sub?
,batch_normalization_41/AssignMovingAvg_1/mulMul0batch_normalization_41/AssignMovingAvg_1/sub:z:07batch_normalization_41/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_41/AssignMovingAvg_1/mul?
(batch_normalization_41/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_41_assignmovingavg_1_readvariableop_resource0batch_normalization_41/AssignMovingAvg_1/mul:z:08^batch_normalization_41/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_41/AssignMovingAvg_1?
&batch_normalization_41/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_41/batchnorm/add/y?
$batch_normalization_41/batchnorm/addAddV21batch_normalization_41/moments/Squeeze_1:output:0/batch_normalization_41/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_41/batchnorm/add?
&batch_normalization_41/batchnorm/RsqrtRsqrt(batch_normalization_41/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_41/batchnorm/Rsqrt?
3batch_normalization_41/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_41_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_41/batchnorm/mul/ReadVariableOp?
$batch_normalization_41/batchnorm/mulMul*batch_normalization_41/batchnorm/Rsqrt:y:0;batch_normalization_41/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_41/batchnorm/mul?
&batch_normalization_41/batchnorm/mul_1Mulconv3d_16/Relu:activations:0(batch_normalization_41/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
   2(
&batch_normalization_41/batchnorm/mul_1?
&batch_normalization_41/batchnorm/mul_2Mul/batch_normalization_41/moments/Squeeze:output:0(batch_normalization_41/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_41/batchnorm/mul_2?
/batch_normalization_41/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_41_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_41/batchnorm/ReadVariableOp?
$batch_normalization_41/batchnorm/subSub7batch_normalization_41/batchnorm/ReadVariableOp:value:0*batch_normalization_41/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_41/batchnorm/sub?
&batch_normalization_41/batchnorm/add_1AddV2*batch_normalization_41/batchnorm/mul_1:z:0(batch_normalization_41/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
   2(
&batch_normalization_41/batchnorm/add_1?
max_pooling3d_31/MaxPool3D	MaxPool3D*batch_normalization_41/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_31/MaxPool3D?
conv3d_17/Conv3D/ReadVariableOpReadVariableOp(conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02!
conv3d_17/Conv3D/ReadVariableOp?
conv3d_17/Conv3DConv3D#max_pooling3d_31/MaxPool3D:output:0'conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
conv3d_17/Conv3D?
 conv3d_17/BiasAdd/ReadVariableOpReadVariableOp)conv3d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_17/BiasAdd/ReadVariableOp?
conv3d_17/BiasAddBiasAddconv3d_17/Conv3D:output:0(conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_17/BiasAdd?
conv3d_17/ReluReluconv3d_17/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_17/Relu?
5batch_normalization_42/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_42/moments/mean/reduction_indices?
#batch_normalization_42/moments/meanMeanconv3d_17/Relu:activations:0>batch_normalization_42/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2%
#batch_normalization_42/moments/mean?
+batch_normalization_42/moments/StopGradientStopGradient,batch_normalization_42/moments/mean:output:0*
T0**
_output_shapes
: 2-
+batch_normalization_42/moments/StopGradient?
0batch_normalization_42/moments/SquaredDifferenceSquaredDifferenceconv3d_17/Relu:activations:04batch_normalization_42/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
 22
0batch_normalization_42/moments/SquaredDifference?
9batch_normalization_42/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_42/moments/variance/reduction_indices?
'batch_normalization_42/moments/varianceMean4batch_normalization_42/moments/SquaredDifference:z:0Bbatch_normalization_42/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2)
'batch_normalization_42/moments/variance?
&batch_normalization_42/moments/SqueezeSqueeze,batch_normalization_42/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_42/moments/Squeeze?
(batch_normalization_42/moments/Squeeze_1Squeeze0batch_normalization_42/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_42/moments/Squeeze_1?
,batch_normalization_42/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_42/AssignMovingAvg/decay?
5batch_normalization_42/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_42_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_42/AssignMovingAvg/ReadVariableOp?
*batch_normalization_42/AssignMovingAvg/subSub=batch_normalization_42/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_42/moments/Squeeze:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_42/AssignMovingAvg/sub?
*batch_normalization_42/AssignMovingAvg/mulMul.batch_normalization_42/AssignMovingAvg/sub:z:05batch_normalization_42/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_42/AssignMovingAvg/mul?
&batch_normalization_42/AssignMovingAvgAssignSubVariableOp>batch_normalization_42_assignmovingavg_readvariableop_resource.batch_normalization_42/AssignMovingAvg/mul:z:06^batch_normalization_42/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_42/AssignMovingAvg?
.batch_normalization_42/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_42/AssignMovingAvg_1/decay?
7batch_normalization_42/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_42_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_42/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_42/AssignMovingAvg_1/subSub?batch_normalization_42/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_42/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_42/AssignMovingAvg_1/sub?
,batch_normalization_42/AssignMovingAvg_1/mulMul0batch_normalization_42/AssignMovingAvg_1/sub:z:07batch_normalization_42/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_42/AssignMovingAvg_1/mul?
(batch_normalization_42/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_42_assignmovingavg_1_readvariableop_resource0batch_normalization_42/AssignMovingAvg_1/mul:z:08^batch_normalization_42/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_42/AssignMovingAvg_1?
&batch_normalization_42/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_42/batchnorm/add/y?
$batch_normalization_42/batchnorm/addAddV21batch_normalization_42/moments/Squeeze_1:output:0/batch_normalization_42/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_42/batchnorm/add?
&batch_normalization_42/batchnorm/RsqrtRsqrt(batch_normalization_42/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_42/batchnorm/Rsqrt?
3batch_normalization_42/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_42_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_42/batchnorm/mul/ReadVariableOp?
$batch_normalization_42/batchnorm/mulMul*batch_normalization_42/batchnorm/Rsqrt:y:0;batch_normalization_42/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_42/batchnorm/mul?
&batch_normalization_42/batchnorm/mul_1Mulconv3d_17/Relu:activations:0(batch_normalization_42/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_42/batchnorm/mul_1?
&batch_normalization_42/batchnorm/mul_2Mul/batch_normalization_42/moments/Squeeze:output:0(batch_normalization_42/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_42/batchnorm/mul_2?
/batch_normalization_42/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_42_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_42/batchnorm/ReadVariableOp?
$batch_normalization_42/batchnorm/subSub7batch_normalization_42/batchnorm/ReadVariableOp:value:0*batch_normalization_42/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_42/batchnorm/sub?
&batch_normalization_42/batchnorm/add_1AddV2*batch_normalization_42/batchnorm/mul_1:z:0(batch_normalization_42/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_42/batchnorm/add_1?
max_pooling3d_32/MaxPool3D	MaxPool3D*batch_normalization_42/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_32/MaxPool3D?
conv3d_18/Conv3D/ReadVariableOpReadVariableOp(conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02!
conv3d_18/Conv3D/ReadVariableOp?
conv3d_18/Conv3DConv3D#max_pooling3d_32/MaxPool3D:output:0'conv3d_18/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
conv3d_18/Conv3D?
 conv3d_18/BiasAdd/ReadVariableOpReadVariableOp)conv3d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_18/BiasAdd/ReadVariableOp?
conv3d_18/BiasAddBiasAddconv3d_18/Conv3D:output:0(conv3d_18/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_18/BiasAdd?
conv3d_18/ReluReluconv3d_18/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_18/Relu?
5batch_normalization_43/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_43/moments/mean/reduction_indices?
#batch_normalization_43/moments/meanMeanconv3d_18/Relu:activations:0>batch_normalization_43/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2%
#batch_normalization_43/moments/mean?
+batch_normalization_43/moments/StopGradientStopGradient,batch_normalization_43/moments/mean:output:0*
T0**
_output_shapes
: 2-
+batch_normalization_43/moments/StopGradient?
0batch_normalization_43/moments/SquaredDifferenceSquaredDifferenceconv3d_18/Relu:activations:04batch_normalization_43/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
 22
0batch_normalization_43/moments/SquaredDifference?
9batch_normalization_43/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_43/moments/variance/reduction_indices?
'batch_normalization_43/moments/varianceMean4batch_normalization_43/moments/SquaredDifference:z:0Bbatch_normalization_43/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2)
'batch_normalization_43/moments/variance?
&batch_normalization_43/moments/SqueezeSqueeze,batch_normalization_43/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_43/moments/Squeeze?
(batch_normalization_43/moments/Squeeze_1Squeeze0batch_normalization_43/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_43/moments/Squeeze_1?
,batch_normalization_43/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_43/AssignMovingAvg/decay?
5batch_normalization_43/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_43_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_43/AssignMovingAvg/ReadVariableOp?
*batch_normalization_43/AssignMovingAvg/subSub=batch_normalization_43/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_43/moments/Squeeze:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_43/AssignMovingAvg/sub?
*batch_normalization_43/AssignMovingAvg/mulMul.batch_normalization_43/AssignMovingAvg/sub:z:05batch_normalization_43/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_43/AssignMovingAvg/mul?
&batch_normalization_43/AssignMovingAvgAssignSubVariableOp>batch_normalization_43_assignmovingavg_readvariableop_resource.batch_normalization_43/AssignMovingAvg/mul:z:06^batch_normalization_43/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_43/AssignMovingAvg?
.batch_normalization_43/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_43/AssignMovingAvg_1/decay?
7batch_normalization_43/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_43_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_43/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_43/AssignMovingAvg_1/subSub?batch_normalization_43/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_43/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_43/AssignMovingAvg_1/sub?
,batch_normalization_43/AssignMovingAvg_1/mulMul0batch_normalization_43/AssignMovingAvg_1/sub:z:07batch_normalization_43/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_43/AssignMovingAvg_1/mul?
(batch_normalization_43/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_43_assignmovingavg_1_readvariableop_resource0batch_normalization_43/AssignMovingAvg_1/mul:z:08^batch_normalization_43/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_43/AssignMovingAvg_1?
&batch_normalization_43/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_43/batchnorm/add/y?
$batch_normalization_43/batchnorm/addAddV21batch_normalization_43/moments/Squeeze_1:output:0/batch_normalization_43/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_43/batchnorm/add?
&batch_normalization_43/batchnorm/RsqrtRsqrt(batch_normalization_43/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_43/batchnorm/Rsqrt?
3batch_normalization_43/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_43_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_43/batchnorm/mul/ReadVariableOp?
$batch_normalization_43/batchnorm/mulMul*batch_normalization_43/batchnorm/Rsqrt:y:0;batch_normalization_43/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_43/batchnorm/mul?
&batch_normalization_43/batchnorm/mul_1Mulconv3d_18/Relu:activations:0(batch_normalization_43/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_43/batchnorm/mul_1?
&batch_normalization_43/batchnorm/mul_2Mul/batch_normalization_43/moments/Squeeze:output:0(batch_normalization_43/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_43/batchnorm/mul_2?
/batch_normalization_43/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_43_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_43/batchnorm/ReadVariableOp?
$batch_normalization_43/batchnorm/subSub7batch_normalization_43/batchnorm/ReadVariableOp:value:0*batch_normalization_43/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_43/batchnorm/sub?
&batch_normalization_43/batchnorm/add_1AddV2*batch_normalization_43/batchnorm/mul_1:z:0(batch_normalization_43/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_43/batchnorm/add_1?
max_pooling3d_33/MaxPool3D	MaxPool3D*batch_normalization_43/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_33/MaxPool3D?
conv3d_19/Conv3D/ReadVariableOpReadVariableOp(conv3d_19_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02!
conv3d_19/Conv3D/ReadVariableOp?
conv3d_19/Conv3DConv3D#max_pooling3d_33/MaxPool3D:output:0'conv3d_19/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
conv3d_19/Conv3D?
 conv3d_19/BiasAdd/ReadVariableOpReadVariableOp)conv3d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_19/BiasAdd/ReadVariableOp?
conv3d_19/BiasAddBiasAddconv3d_19/Conv3D:output:0(conv3d_19/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_19/BiasAdd?
conv3d_19/ReluReluconv3d_19/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
conv3d_19/Relu?
5batch_normalization_44/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             27
5batch_normalization_44/moments/mean/reduction_indices?
#batch_normalization_44/moments/meanMeanconv3d_19/Relu:activations:0>batch_normalization_44/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2%
#batch_normalization_44/moments/mean?
+batch_normalization_44/moments/StopGradientStopGradient,batch_normalization_44/moments/mean:output:0*
T0**
_output_shapes
: 2-
+batch_normalization_44/moments/StopGradient?
0batch_normalization_44/moments/SquaredDifferenceSquaredDifferenceconv3d_19/Relu:activations:04batch_normalization_44/moments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
 22
0batch_normalization_44/moments/SquaredDifference?
9batch_normalization_44/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9batch_normalization_44/moments/variance/reduction_indices?
'batch_normalization_44/moments/varianceMean4batch_normalization_44/moments/SquaredDifference:z:0Bbatch_normalization_44/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2)
'batch_normalization_44/moments/variance?
&batch_normalization_44/moments/SqueezeSqueeze,batch_normalization_44/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_44/moments/Squeeze?
(batch_normalization_44/moments/Squeeze_1Squeeze0batch_normalization_44/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_44/moments/Squeeze_1?
,batch_normalization_44/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_44/AssignMovingAvg/decay?
5batch_normalization_44/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_44_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_44/AssignMovingAvg/ReadVariableOp?
*batch_normalization_44/AssignMovingAvg/subSub=batch_normalization_44/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_44/moments/Squeeze:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_44/AssignMovingAvg/sub?
*batch_normalization_44/AssignMovingAvg/mulMul.batch_normalization_44/AssignMovingAvg/sub:z:05batch_normalization_44/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_44/AssignMovingAvg/mul?
&batch_normalization_44/AssignMovingAvgAssignSubVariableOp>batch_normalization_44_assignmovingavg_readvariableop_resource.batch_normalization_44/AssignMovingAvg/mul:z:06^batch_normalization_44/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_44/AssignMovingAvg?
.batch_normalization_44/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_44/AssignMovingAvg_1/decay?
7batch_normalization_44/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_44_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_44/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_44/AssignMovingAvg_1/subSub?batch_normalization_44/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_44/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_44/AssignMovingAvg_1/sub?
,batch_normalization_44/AssignMovingAvg_1/mulMul0batch_normalization_44/AssignMovingAvg_1/sub:z:07batch_normalization_44/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_44/AssignMovingAvg_1/mul?
(batch_normalization_44/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_44_assignmovingavg_1_readvariableop_resource0batch_normalization_44/AssignMovingAvg_1/mul:z:08^batch_normalization_44/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_44/AssignMovingAvg_1?
&batch_normalization_44/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_44/batchnorm/add/y?
$batch_normalization_44/batchnorm/addAddV21batch_normalization_44/moments/Squeeze_1:output:0/batch_normalization_44/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_44/batchnorm/add?
&batch_normalization_44/batchnorm/RsqrtRsqrt(batch_normalization_44/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_44/batchnorm/Rsqrt?
3batch_normalization_44/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_44_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_44/batchnorm/mul/ReadVariableOp?
$batch_normalization_44/batchnorm/mulMul*batch_normalization_44/batchnorm/Rsqrt:y:0;batch_normalization_44/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_44/batchnorm/mul?
&batch_normalization_44/batchnorm/mul_1Mulconv3d_19/Relu:activations:0(batch_normalization_44/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_44/batchnorm/mul_1?
&batch_normalization_44/batchnorm/mul_2Mul/batch_normalization_44/moments/Squeeze:output:0(batch_normalization_44/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_44/batchnorm/mul_2?
/batch_normalization_44/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_44_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_44/batchnorm/ReadVariableOp?
$batch_normalization_44/batchnorm/subSub7batch_normalization_44/batchnorm/ReadVariableOp:value:0*batch_normalization_44/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_44/batchnorm/sub?
&batch_normalization_44/batchnorm/add_1AddV2*batch_normalization_44/batchnorm/mul_1:z:0(batch_normalization_44/batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2(
&batch_normalization_44/batchnorm/add_1?
max_pooling3d_34/MaxPool3D	MaxPool3D*batch_normalization_44/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:?????????
 *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_34/MaxPool3Ds
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_8/Const?
flatten_8/ReshapeReshape#max_pooling3d_34/MaxPool3D:output:0flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????
2
flatten_8/Reshapey
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_16/dropout/Const?
dropout_16/dropout/MulMulflatten_8/Reshape:output:0!dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:??????????
2
dropout_16/dropout/Mul~
dropout_16/dropout/ShapeShapeflatten_8/Reshape:output:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape?
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????
*
dtype021
/dropout_16/dropout/random_uniform/RandomUniform?
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2#
!dropout_16/dropout/GreaterEqual/y?
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????
2!
dropout_16/dropout/GreaterEqual?
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????
2
dropout_16/dropout/Cast?
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????
2
dropout_16/dropout/Mul_1?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMuldropout_16/dropout/Mul_1:z:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/BiasAddt
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_16/Reluy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_17/dropout/Const?
dropout_17/dropout/MulMuldense_16/Relu:activations:0!dropout_17/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_17/dropout/Mul
dropout_17/dropout/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape?
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout_17/dropout/random_uniform/RandomUniform?
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_17/dropout/GreaterEqual/y?
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout_17/dropout/GreaterEqual?
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_17/dropout/Cast?
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_17/dropout/Mul_1?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Softmax?
2conv3d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_16/kernel/Regularizer/SquareSquare:conv3d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_16/kernel/Regularizer/Square?
"conv3d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_16/kernel/Regularizer/Const?
 conv3d_16/kernel/Regularizer/SumSum'conv3d_16/kernel/Regularizer/Square:y:0+conv3d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/Sum?
"conv3d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_16/kernel/Regularizer/mul/x?
 conv3d_16/kernel/Regularizer/mulMul+conv3d_16/kernel/Regularizer/mul/x:output:0)conv3d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/mul?
2conv3d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_17/kernel/Regularizer/SquareSquare:conv3d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_17/kernel/Regularizer/Square?
"conv3d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_17/kernel/Regularizer/Const?
 conv3d_17/kernel/Regularizer/SumSum'conv3d_17/kernel/Regularizer/Square:y:0+conv3d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/Sum?
"conv3d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_17/kernel/Regularizer/mul/x?
 conv3d_17/kernel/Regularizer/mulMul+conv3d_17/kernel/Regularizer/mul/x:output:0)conv3d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/mul?
2conv3d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_18/kernel/Regularizer/SquareSquare:conv3d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_18/kernel/Regularizer/Square?
"conv3d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_18/kernel/Regularizer/Const?
 conv3d_18/kernel/Regularizer/SumSum'conv3d_18/kernel/Regularizer/Square:y:0+conv3d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/Sum?
"conv3d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_18/kernel/Regularizer/mul/x?
 conv3d_18/kernel/Regularizer/mulMul+conv3d_18/kernel/Regularizer/mul/x:output:0)conv3d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/mul?
2conv3d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv3d_19_conv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_19/kernel/Regularizer/SquareSquare:conv3d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_19/kernel/Regularizer/Square?
"conv3d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_19/kernel/Regularizer/Const?
 conv3d_19/kernel/Regularizer/SumSum'conv3d_19/kernel/Regularizer/Square:y:0+conv3d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/Sum?
"conv3d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_19/kernel/Regularizer/mul/x?
 conv3d_19/kernel/Regularizer/mulMul+conv3d_19/kernel/Regularizer/mul/x:output:0)conv3d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/mul?
IdentityIdentitydense_17/Softmax:softmax:0'^batch_normalization_40/AssignMovingAvg6^batch_normalization_40/AssignMovingAvg/ReadVariableOp)^batch_normalization_40/AssignMovingAvg_18^batch_normalization_40/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_40/batchnorm/ReadVariableOp4^batch_normalization_40/batchnorm/mul/ReadVariableOp'^batch_normalization_41/AssignMovingAvg6^batch_normalization_41/AssignMovingAvg/ReadVariableOp)^batch_normalization_41/AssignMovingAvg_18^batch_normalization_41/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_41/batchnorm/ReadVariableOp4^batch_normalization_41/batchnorm/mul/ReadVariableOp'^batch_normalization_42/AssignMovingAvg6^batch_normalization_42/AssignMovingAvg/ReadVariableOp)^batch_normalization_42/AssignMovingAvg_18^batch_normalization_42/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_42/batchnorm/ReadVariableOp4^batch_normalization_42/batchnorm/mul/ReadVariableOp'^batch_normalization_43/AssignMovingAvg6^batch_normalization_43/AssignMovingAvg/ReadVariableOp)^batch_normalization_43/AssignMovingAvg_18^batch_normalization_43/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_43/batchnorm/ReadVariableOp4^batch_normalization_43/batchnorm/mul/ReadVariableOp'^batch_normalization_44/AssignMovingAvg6^batch_normalization_44/AssignMovingAvg/ReadVariableOp)^batch_normalization_44/AssignMovingAvg_18^batch_normalization_44/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_44/batchnorm/ReadVariableOp4^batch_normalization_44/batchnorm/mul/ReadVariableOp!^conv3d_15/BiasAdd/ReadVariableOp ^conv3d_15/Conv3D/ReadVariableOp!^conv3d_16/BiasAdd/ReadVariableOp ^conv3d_16/Conv3D/ReadVariableOp3^conv3d_16/kernel/Regularizer/Square/ReadVariableOp!^conv3d_17/BiasAdd/ReadVariableOp ^conv3d_17/Conv3D/ReadVariableOp3^conv3d_17/kernel/Regularizer/Square/ReadVariableOp!^conv3d_18/BiasAdd/ReadVariableOp ^conv3d_18/Conv3D/ReadVariableOp3^conv3d_18/kernel/Regularizer/Square/ReadVariableOp!^conv3d_19/BiasAdd/ReadVariableOp ^conv3d_19/Conv3D/ReadVariableOp3^conv3d_19/kernel/Regularizer/Square/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_40/AssignMovingAvg&batch_normalization_40/AssignMovingAvg2n
5batch_normalization_40/AssignMovingAvg/ReadVariableOp5batch_normalization_40/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_40/AssignMovingAvg_1(batch_normalization_40/AssignMovingAvg_12r
7batch_normalization_40/AssignMovingAvg_1/ReadVariableOp7batch_normalization_40/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_40/batchnorm/ReadVariableOp/batch_normalization_40/batchnorm/ReadVariableOp2j
3batch_normalization_40/batchnorm/mul/ReadVariableOp3batch_normalization_40/batchnorm/mul/ReadVariableOp2P
&batch_normalization_41/AssignMovingAvg&batch_normalization_41/AssignMovingAvg2n
5batch_normalization_41/AssignMovingAvg/ReadVariableOp5batch_normalization_41/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_41/AssignMovingAvg_1(batch_normalization_41/AssignMovingAvg_12r
7batch_normalization_41/AssignMovingAvg_1/ReadVariableOp7batch_normalization_41/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_41/batchnorm/ReadVariableOp/batch_normalization_41/batchnorm/ReadVariableOp2j
3batch_normalization_41/batchnorm/mul/ReadVariableOp3batch_normalization_41/batchnorm/mul/ReadVariableOp2P
&batch_normalization_42/AssignMovingAvg&batch_normalization_42/AssignMovingAvg2n
5batch_normalization_42/AssignMovingAvg/ReadVariableOp5batch_normalization_42/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_42/AssignMovingAvg_1(batch_normalization_42/AssignMovingAvg_12r
7batch_normalization_42/AssignMovingAvg_1/ReadVariableOp7batch_normalization_42/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_42/batchnorm/ReadVariableOp/batch_normalization_42/batchnorm/ReadVariableOp2j
3batch_normalization_42/batchnorm/mul/ReadVariableOp3batch_normalization_42/batchnorm/mul/ReadVariableOp2P
&batch_normalization_43/AssignMovingAvg&batch_normalization_43/AssignMovingAvg2n
5batch_normalization_43/AssignMovingAvg/ReadVariableOp5batch_normalization_43/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_43/AssignMovingAvg_1(batch_normalization_43/AssignMovingAvg_12r
7batch_normalization_43/AssignMovingAvg_1/ReadVariableOp7batch_normalization_43/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_43/batchnorm/ReadVariableOp/batch_normalization_43/batchnorm/ReadVariableOp2j
3batch_normalization_43/batchnorm/mul/ReadVariableOp3batch_normalization_43/batchnorm/mul/ReadVariableOp2P
&batch_normalization_44/AssignMovingAvg&batch_normalization_44/AssignMovingAvg2n
5batch_normalization_44/AssignMovingAvg/ReadVariableOp5batch_normalization_44/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_44/AssignMovingAvg_1(batch_normalization_44/AssignMovingAvg_12r
7batch_normalization_44/AssignMovingAvg_1/ReadVariableOp7batch_normalization_44/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_44/batchnorm/ReadVariableOp/batch_normalization_44/batchnorm/ReadVariableOp2j
3batch_normalization_44/batchnorm/mul/ReadVariableOp3batch_normalization_44/batchnorm/mul/ReadVariableOp2D
 conv3d_15/BiasAdd/ReadVariableOp conv3d_15/BiasAdd/ReadVariableOp2B
conv3d_15/Conv3D/ReadVariableOpconv3d_15/Conv3D/ReadVariableOp2D
 conv3d_16/BiasAdd/ReadVariableOp conv3d_16/BiasAdd/ReadVariableOp2B
conv3d_16/Conv3D/ReadVariableOpconv3d_16/Conv3D/ReadVariableOp2h
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_17/BiasAdd/ReadVariableOp conv3d_17/BiasAdd/ReadVariableOp2B
conv3d_17/Conv3D/ReadVariableOpconv3d_17/Conv3D/ReadVariableOp2h
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_18/BiasAdd/ReadVariableOp conv3d_18/BiasAdd/ReadVariableOp2B
conv3d_18/Conv3D/ReadVariableOpconv3d_18/Conv3D/ReadVariableOp2h
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_19/BiasAdd/ReadVariableOp conv3d_19/BiasAdd/ReadVariableOp2B
conv3d_19/Conv3D/ReadVariableOpconv3d_19/Conv3D/ReadVariableOp2h
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
@@
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_40_layer_call_fn_421110

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4181572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421415

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
   2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
   2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
   : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
   
 
_user_specified_nameinputs
?,
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_418913

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
d
+__inference_dropout_16_layer_call_fn_422046

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_4194842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421799

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_420178
conv3d_15_input%
unknown:
 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: '
	unknown_5:
  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: (

unknown_11:
  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: (

unknown_17:
  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: (

unknown_23:
  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29:
?
?

unknown_30:	?

unknown_31:	?

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
 !"*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_4200342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
3
_output_shapes!
:?????????
@@
)
_user_specified_nameconv3d_15_input
?
?
E__inference_conv3d_17_layer_call_and_return_conditional_losses_421481

inputs<
conv3d_readvariableop_resource:
  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
Relu?
2conv3d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_17/kernel/Regularizer/SquareSquare:conv3d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_17/kernel/Regularizer/Square?
"conv3d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_17/kernel/Regularizer/Const?
 conv3d_17/kernel/Regularizer/SumSum'conv3d_17/kernel/Regularizer/Square:y:0+conv3d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/Sum?
"conv3d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_17/kernel/Regularizer/mul/x?
 conv3d_17/kernel/Regularizer/mulMul+conv3d_17/kernel/Regularizer/mul/x:output:0)conv3d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp3^conv3d_17/kernel/Regularizer/Square/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2h
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2conv3d_17/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
?
*__inference_conv3d_16_layer_call_fn_421272

inputs%
unknown:
  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_4190742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
   
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_40_layer_call_fn_421149

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_4198222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
@@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
@@ 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_43_layer_call_fn_421725

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_4196122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_419451

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_419752

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
   2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
   2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
   2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
   : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
   
 
_user_specified_nameinputs
?
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_419484

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????
2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????
2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????
2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_418505

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_422152Y
;conv3d_17_kernel_regularizer_square_readvariableop_resource:
  
identity??2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
2conv3d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv3d_17_kernel_regularizer_square_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_17/kernel/Regularizer/SquareSquare:conv3d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_17/kernel/Regularizer/Square?
"conv3d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_17/kernel/Regularizer/Const?
 conv3d_17/kernel/Regularizer/SumSum'conv3d_17/kernel/Regularizer/Square:y:0+conv3d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/Sum?
"conv3d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_17/kernel/Regularizer/mul/x?
 conv3d_17/kernel/Regularizer/mulMul+conv3d_17/kernel/Regularizer/mul/x:output:0)conv3d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/mul?
IdentityIdentity$conv3d_17/kernel/Regularizer/mul:z:03^conv3d_17/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2conv3d_17/kernel/Regularizer/Square/ReadVariableOp
?
?
E__inference_conv3d_16_layer_call_and_return_conditional_losses_421289

inputs<
conv3d_readvariableop_resource:
  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
   *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
   2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
   2
Relu?
2conv3d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_16/kernel/Regularizer/SquareSquare:conv3d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_16/kernel/Regularizer/Square?
"conv3d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_16/kernel/Regularizer/Const?
 conv3d_16/kernel/Regularizer/SumSum'conv3d_16/kernel/Regularizer/Square:y:0+conv3d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/Sum?
"conv3d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_16/kernel/Regularizer/mul/x?
 conv3d_16/kernel/Regularizer/mulMul+conv3d_16/kernel/Regularizer/mul/x:output:0)conv3d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp3^conv3d_16/kernel/Regularizer/Square/ReadVariableOp*
T0*3
_output_shapes!
:?????????
   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2h
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2conv3d_16/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:?????????
   
 
_user_specified_nameinputs
ّ
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_419350

inputs.
conv3d_15_419022:
 
conv3d_15_419024: +
batch_normalization_40_419047: +
batch_normalization_40_419049: +
batch_normalization_40_419051: +
batch_normalization_40_419053: .
conv3d_16_419075:
  
conv3d_16_419077: +
batch_normalization_41_419100: +
batch_normalization_41_419102: +
batch_normalization_41_419104: +
batch_normalization_41_419106: .
conv3d_17_419128:
  
conv3d_17_419130: +
batch_normalization_42_419153: +
batch_normalization_42_419155: +
batch_normalization_42_419157: +
batch_normalization_42_419159: .
conv3d_18_419181:
  
conv3d_18_419183: +
batch_normalization_43_419206: +
batch_normalization_43_419208: +
batch_normalization_43_419210: +
batch_normalization_43_419212: .
conv3d_19_419234:
  
conv3d_19_419236: +
batch_normalization_44_419259: +
batch_normalization_44_419261: +
batch_normalization_44_419263: +
batch_normalization_44_419265: #
dense_16_419296:
?
?
dense_16_419298:	?"
dense_17_419320:	?
dense_17_419322:
identity??.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?!conv3d_15/StatefulPartitionedCall?!conv3d_16/StatefulPartitionedCall?2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?!conv3d_17/StatefulPartitionedCall?2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?!conv3d_18/StatefulPartitionedCall?2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?!conv3d_19/StatefulPartitionedCall?2conv3d_19/kernel/Regularizer/Square/ReadVariableOp? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_15_419022conv3d_15_419024*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_4190212#
!conv3d_15/StatefulPartitionedCall?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0batch_normalization_40_419047batch_normalization_40_419049batch_normalization_40_419051batch_normalization_40_419053*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_41904620
.batch_normalization_40/StatefulPartitionedCall?
 max_pooling3d_30/PartitionedCallPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_30_layer_call_and_return_conditional_losses_4183012"
 max_pooling3d_30/PartitionedCall?
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_30/PartitionedCall:output:0conv3d_16_419075conv3d_16_419077*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_4190742#
!conv3d_16/StatefulPartitionedCall?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0batch_normalization_41_419100batch_normalization_41_419102batch_normalization_41_419104batch_normalization_41_419106*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_41909920
.batch_normalization_41/StatefulPartitionedCall?
 max_pooling3d_31/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_31_layer_call_and_return_conditional_losses_4184752"
 max_pooling3d_31/PartitionedCall?
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_31/PartitionedCall:output:0conv3d_17_419128conv3d_17_419130*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_4191272#
!conv3d_17/StatefulPartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0batch_normalization_42_419153batch_normalization_42_419155batch_normalization_42_419157batch_normalization_42_419159*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_41915220
.batch_normalization_42/StatefulPartitionedCall?
 max_pooling3d_32/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_32_layer_call_and_return_conditional_losses_4186492"
 max_pooling3d_32/PartitionedCall?
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_32/PartitionedCall:output:0conv3d_18_419181conv3d_18_419183*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_18_layer_call_and_return_conditional_losses_4191802#
!conv3d_18/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0batch_normalization_43_419206batch_normalization_43_419208batch_normalization_43_419210batch_normalization_43_419212*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_41920520
.batch_normalization_43/StatefulPartitionedCall?
 max_pooling3d_33/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_33_layer_call_and_return_conditional_losses_4188232"
 max_pooling3d_33/PartitionedCall?
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_33/PartitionedCall:output:0conv3d_19_419234conv3d_19_419236*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_19_layer_call_and_return_conditional_losses_4192332#
!conv3d_19/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0batch_normalization_44_419259batch_normalization_44_419261batch_normalization_44_419263batch_normalization_44_419265*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_41925820
.batch_normalization_44/StatefulPartitionedCall?
 max_pooling3d_34/PartitionedCallPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_34_layer_call_and_return_conditional_losses_4189972"
 max_pooling3d_34/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall)max_pooling3d_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_4192752
flatten_8/PartitionedCall?
dropout_16/PartitionedCallPartitionedCall"flatten_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_4192822
dropout_16/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_16_419296dense_16_419298*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_4192952"
 dense_16/StatefulPartitionedCall?
dropout_17/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_4193062
dropout_17/PartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_17_419320dense_17_419322*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_4193192"
 dense_17/StatefulPartitionedCall?
2conv3d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_16_419075**
_output_shapes
:
  *
dtype024
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_16/kernel/Regularizer/SquareSquare:conv3d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_16/kernel/Regularizer/Square?
"conv3d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_16/kernel/Regularizer/Const?
 conv3d_16/kernel/Regularizer/SumSum'conv3d_16/kernel/Regularizer/Square:y:0+conv3d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/Sum?
"conv3d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_16/kernel/Regularizer/mul/x?
 conv3d_16/kernel/Regularizer/mulMul+conv3d_16/kernel/Regularizer/mul/x:output:0)conv3d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/mul?
2conv3d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_17_419128**
_output_shapes
:
  *
dtype024
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_17/kernel/Regularizer/SquareSquare:conv3d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_17/kernel/Regularizer/Square?
"conv3d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_17/kernel/Regularizer/Const?
 conv3d_17/kernel/Regularizer/SumSum'conv3d_17/kernel/Regularizer/Square:y:0+conv3d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/Sum?
"conv3d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_17/kernel/Regularizer/mul/x?
 conv3d_17/kernel/Regularizer/mulMul+conv3d_17/kernel/Regularizer/mul/x:output:0)conv3d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/mul?
2conv3d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_18_419181**
_output_shapes
:
  *
dtype024
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_18/kernel/Regularizer/SquareSquare:conv3d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_18/kernel/Regularizer/Square?
"conv3d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_18/kernel/Regularizer/Const?
 conv3d_18/kernel/Regularizer/SumSum'conv3d_18/kernel/Regularizer/Square:y:0+conv3d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/Sum?
"conv3d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_18/kernel/Regularizer/mul/x?
 conv3d_18/kernel/Regularizer/mulMul+conv3d_18/kernel/Regularizer/mul/x:output:0)conv3d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/mul?
2conv3d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_19_419234**
_output_shapes
:
  *
dtype024
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_19/kernel/Regularizer/SquareSquare:conv3d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_19/kernel/Regularizer/Square?
"conv3d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_19/kernel/Regularizer/Const?
 conv3d_19/kernel/Regularizer/SumSum'conv3d_19/kernel/Regularizer/Square:y:0+conv3d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/Sum?
"conv3d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_19/kernel/Regularizer/mul/x?
 conv3d_19/kernel/Regularizer/mulMul+conv3d_19/kernel/Regularizer/mul/x:output:0)conv3d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/mul?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall3^conv3d_16/kernel/Regularizer/Square/ReadVariableOp"^conv3d_17/StatefulPartitionedCall3^conv3d_17/kernel/Regularizer/Square/ReadVariableOp"^conv3d_18/StatefulPartitionedCall3^conv3d_18/kernel/Regularizer/Square/ReadVariableOp"^conv3d_19/StatefulPartitionedCall3^conv3d_19/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2h
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2h
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_18/StatefulPartitionedCall!conv3d_18/StatefulPartitionedCall2h
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_19/StatefulPartitionedCall!conv3d_19/StatefulPartitionedCall2h
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
@@
 
_user_specified_nameinputs
?
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_422098

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ܔ
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_420034

inputs.
conv3d_15_419921:
 
conv3d_15_419923: +
batch_normalization_40_419926: +
batch_normalization_40_419928: +
batch_normalization_40_419930: +
batch_normalization_40_419932: .
conv3d_16_419936:
  
conv3d_16_419938: +
batch_normalization_41_419941: +
batch_normalization_41_419943: +
batch_normalization_41_419945: +
batch_normalization_41_419947: .
conv3d_17_419951:
  
conv3d_17_419953: +
batch_normalization_42_419956: +
batch_normalization_42_419958: +
batch_normalization_42_419960: +
batch_normalization_42_419962: .
conv3d_18_419966:
  
conv3d_18_419968: +
batch_normalization_43_419971: +
batch_normalization_43_419973: +
batch_normalization_43_419975: +
batch_normalization_43_419977: .
conv3d_19_419981:
  
conv3d_19_419983: +
batch_normalization_44_419986: +
batch_normalization_44_419988: +
batch_normalization_44_419990: +
batch_normalization_44_419992: #
dense_16_419998:
?
?
dense_16_420000:	?"
dense_17_420004:	?
dense_17_420006:
identity??.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?!conv3d_15/StatefulPartitionedCall?!conv3d_16/StatefulPartitionedCall?2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?!conv3d_17/StatefulPartitionedCall?2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?!conv3d_18/StatefulPartitionedCall?2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?!conv3d_19/StatefulPartitionedCall?2conv3d_19/kernel/Regularizer/Square/ReadVariableOp? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?"dropout_17/StatefulPartitionedCall?
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_15_419921conv3d_15_419923*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_4190212#
!conv3d_15/StatefulPartitionedCall?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0batch_normalization_40_419926batch_normalization_40_419928batch_normalization_40_419930batch_normalization_40_419932*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_41982220
.batch_normalization_40/StatefulPartitionedCall?
 max_pooling3d_30/PartitionedCallPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_30_layer_call_and_return_conditional_losses_4183012"
 max_pooling3d_30/PartitionedCall?
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_30/PartitionedCall:output:0conv3d_16_419936conv3d_16_419938*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_4190742#
!conv3d_16/StatefulPartitionedCall?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0batch_normalization_41_419941batch_normalization_41_419943batch_normalization_41_419945batch_normalization_41_419947*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_41975220
.batch_normalization_41/StatefulPartitionedCall?
 max_pooling3d_31/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_31_layer_call_and_return_conditional_losses_4184752"
 max_pooling3d_31/PartitionedCall?
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_31/PartitionedCall:output:0conv3d_17_419951conv3d_17_419953*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_4191272#
!conv3d_17/StatefulPartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0batch_normalization_42_419956batch_normalization_42_419958batch_normalization_42_419960batch_normalization_42_419962*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_41968220
.batch_normalization_42/StatefulPartitionedCall?
 max_pooling3d_32/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_32_layer_call_and_return_conditional_losses_4186492"
 max_pooling3d_32/PartitionedCall?
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_32/PartitionedCall:output:0conv3d_18_419966conv3d_18_419968*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_18_layer_call_and_return_conditional_losses_4191802#
!conv3d_18/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0batch_normalization_43_419971batch_normalization_43_419973batch_normalization_43_419975batch_normalization_43_419977*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_41961220
.batch_normalization_43/StatefulPartitionedCall?
 max_pooling3d_33/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_33_layer_call_and_return_conditional_losses_4188232"
 max_pooling3d_33/PartitionedCall?
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_33/PartitionedCall:output:0conv3d_19_419981conv3d_19_419983*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_19_layer_call_and_return_conditional_losses_4192332#
!conv3d_19/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0batch_normalization_44_419986batch_normalization_44_419988batch_normalization_44_419990batch_normalization_44_419992*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_41954220
.batch_normalization_44/StatefulPartitionedCall?
 max_pooling3d_34/PartitionedCallPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_34_layer_call_and_return_conditional_losses_4189972"
 max_pooling3d_34/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall)max_pooling3d_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_4192752
flatten_8/PartitionedCall?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_4194842$
"dropout_16/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_16_419998dense_16_420000*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_4192952"
 dense_16/StatefulPartitionedCall?
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_4194512$
"dropout_17/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_17_420004dense_17_420006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_4193192"
 dense_17/StatefulPartitionedCall?
2conv3d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_16_419936**
_output_shapes
:
  *
dtype024
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_16/kernel/Regularizer/SquareSquare:conv3d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_16/kernel/Regularizer/Square?
"conv3d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_16/kernel/Regularizer/Const?
 conv3d_16/kernel/Regularizer/SumSum'conv3d_16/kernel/Regularizer/Square:y:0+conv3d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/Sum?
"conv3d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_16/kernel/Regularizer/mul/x?
 conv3d_16/kernel/Regularizer/mulMul+conv3d_16/kernel/Regularizer/mul/x:output:0)conv3d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/mul?
2conv3d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_17_419951**
_output_shapes
:
  *
dtype024
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_17/kernel/Regularizer/SquareSquare:conv3d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_17/kernel/Regularizer/Square?
"conv3d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_17/kernel/Regularizer/Const?
 conv3d_17/kernel/Regularizer/SumSum'conv3d_17/kernel/Regularizer/Square:y:0+conv3d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/Sum?
"conv3d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_17/kernel/Regularizer/mul/x?
 conv3d_17/kernel/Regularizer/mulMul+conv3d_17/kernel/Regularizer/mul/x:output:0)conv3d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/mul?
2conv3d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_18_419966**
_output_shapes
:
  *
dtype024
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_18/kernel/Regularizer/SquareSquare:conv3d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_18/kernel/Regularizer/Square?
"conv3d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_18/kernel/Regularizer/Const?
 conv3d_18/kernel/Regularizer/SumSum'conv3d_18/kernel/Regularizer/Square:y:0+conv3d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/Sum?
"conv3d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_18/kernel/Regularizer/mul/x?
 conv3d_18/kernel/Regularizer/mulMul+conv3d_18/kernel/Regularizer/mul/x:output:0)conv3d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/mul?
2conv3d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_19_419981**
_output_shapes
:
  *
dtype024
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_19/kernel/Regularizer/SquareSquare:conv3d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_19/kernel/Regularizer/Square?
"conv3d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_19/kernel/Regularizer/Const?
 conv3d_19/kernel/Regularizer/SumSum'conv3d_19/kernel/Regularizer/Square:y:0+conv3d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/Sum?
"conv3d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_19/kernel/Regularizer/mul/x?
 conv3d_19/kernel/Regularizer/mulMul+conv3d_19/kernel/Regularizer/mul/x:output:0)conv3d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/mul?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall3^conv3d_16/kernel/Regularizer/Square/ReadVariableOp"^conv3d_17/StatefulPartitionedCall3^conv3d_17/kernel/Regularizer/Square/ReadVariableOp"^conv3d_18/StatefulPartitionedCall3^conv3d_18/kernel/Regularizer/Square/ReadVariableOp"^conv3d_19/StatefulPartitionedCall3^conv3d_19/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2h
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2h
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_18/StatefulPartitionedCall!conv3d_18/StatefulPartitionedCall2h
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_19/StatefulPartitionedCall!conv3d_19/StatefulPartitionedCall2h
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
@@
 
_user_specified_nameinputs
?
?
E__inference_conv3d_16_layer_call_and_return_conditional_losses_419074

inputs<
conv3d_readvariableop_resource:
  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
   *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
   2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
   2
Relu?
2conv3d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_16/kernel/Regularizer/SquareSquare:conv3d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_16/kernel/Regularizer/Square?
"conv3d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_16/kernel/Regularizer/Const?
 conv3d_16/kernel/Regularizer/SumSum'conv3d_16/kernel/Regularizer/Square:y:0+conv3d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/Sum?
"conv3d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_16/kernel/Regularizer/mul/x?
 conv3d_16/kernel/Regularizer/mulMul+conv3d_16/kernel/Regularizer/mul/x:output:0)conv3d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp3^conv3d_16/kernel/Regularizer/Square/ReadVariableOp*
T0*3
_output_shapes!
:?????????
   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2h
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2conv3d_16/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:?????????
   
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_42_layer_call_fn_421520

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_4191522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
?
E__inference_conv3d_18_layer_call_and_return_conditional_losses_419180

inputs<
conv3d_readvariableop_resource:
  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
 2
Relu?
2conv3d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_18/kernel/Regularizer/SquareSquare:conv3d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_18/kernel/Regularizer/Square?
"conv3d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_18/kernel/Regularizer/Const?
 conv3d_18/kernel/Regularizer/SumSum'conv3d_18/kernel/Regularizer/Square:y:0+conv3d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/Sum?
"conv3d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_18/kernel/Regularizer/mul/x?
 conv3d_18/kernel/Regularizer/mulMul+conv3d_18/kernel/Regularizer/mul/x:output:0)conv3d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp3^conv3d_18/kernel/Regularizer/Square/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????
 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2h
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2conv3d_18/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_33_layer_call_fn_418829

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_33_layer_call_and_return_conditional_losses_4188232
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_8_layer_call_fn_422030

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_4192752
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
 :[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?+
?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421449

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
   2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
   2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
   2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
   : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
   
 
_user_specified_nameinputs
??
?N
"__inference__traced_restore_422897
file_prefix?
!assignvariableop_conv3d_15_kernel:
 /
!assignvariableop_1_conv3d_15_bias: =
/assignvariableop_2_batch_normalization_40_gamma: <
.assignvariableop_3_batch_normalization_40_beta: C
5assignvariableop_4_batch_normalization_40_moving_mean: G
9assignvariableop_5_batch_normalization_40_moving_variance: A
#assignvariableop_6_conv3d_16_kernel:
  /
!assignvariableop_7_conv3d_16_bias: =
/assignvariableop_8_batch_normalization_41_gamma: <
.assignvariableop_9_batch_normalization_41_beta: D
6assignvariableop_10_batch_normalization_41_moving_mean: H
:assignvariableop_11_batch_normalization_41_moving_variance: B
$assignvariableop_12_conv3d_17_kernel:
  0
"assignvariableop_13_conv3d_17_bias: >
0assignvariableop_14_batch_normalization_42_gamma: =
/assignvariableop_15_batch_normalization_42_beta: D
6assignvariableop_16_batch_normalization_42_moving_mean: H
:assignvariableop_17_batch_normalization_42_moving_variance: B
$assignvariableop_18_conv3d_18_kernel:
  0
"assignvariableop_19_conv3d_18_bias: >
0assignvariableop_20_batch_normalization_43_gamma: =
/assignvariableop_21_batch_normalization_43_beta: D
6assignvariableop_22_batch_normalization_43_moving_mean: H
:assignvariableop_23_batch_normalization_43_moving_variance: B
$assignvariableop_24_conv3d_19_kernel:
  0
"assignvariableop_25_conv3d_19_bias: >
0assignvariableop_26_batch_normalization_44_gamma: =
/assignvariableop_27_batch_normalization_44_beta: D
6assignvariableop_28_batch_normalization_44_moving_mean: H
:assignvariableop_29_batch_normalization_44_moving_variance: 7
#assignvariableop_30_dense_16_kernel:
?
?0
!assignvariableop_31_dense_16_bias:	?6
#assignvariableop_32_dense_17_kernel:	?/
!assignvariableop_33_dense_17_bias:'
assignvariableop_34_adam_iter:	 )
assignvariableop_35_adam_beta_1: )
assignvariableop_36_adam_beta_2: (
assignvariableop_37_adam_decay: 0
&assignvariableop_38_adam_learning_rate: #
assignvariableop_39_total: #
assignvariableop_40_count: %
assignvariableop_41_total_1: %
assignvariableop_42_count_1: I
+assignvariableop_43_adam_conv3d_15_kernel_m:
 7
)assignvariableop_44_adam_conv3d_15_bias_m: E
7assignvariableop_45_adam_batch_normalization_40_gamma_m: D
6assignvariableop_46_adam_batch_normalization_40_beta_m: I
+assignvariableop_47_adam_conv3d_16_kernel_m:
  7
)assignvariableop_48_adam_conv3d_16_bias_m: E
7assignvariableop_49_adam_batch_normalization_41_gamma_m: D
6assignvariableop_50_adam_batch_normalization_41_beta_m: I
+assignvariableop_51_adam_conv3d_17_kernel_m:
  7
)assignvariableop_52_adam_conv3d_17_bias_m: E
7assignvariableop_53_adam_batch_normalization_42_gamma_m: D
6assignvariableop_54_adam_batch_normalization_42_beta_m: I
+assignvariableop_55_adam_conv3d_18_kernel_m:
  7
)assignvariableop_56_adam_conv3d_18_bias_m: E
7assignvariableop_57_adam_batch_normalization_43_gamma_m: D
6assignvariableop_58_adam_batch_normalization_43_beta_m: I
+assignvariableop_59_adam_conv3d_19_kernel_m:
  7
)assignvariableop_60_adam_conv3d_19_bias_m: E
7assignvariableop_61_adam_batch_normalization_44_gamma_m: D
6assignvariableop_62_adam_batch_normalization_44_beta_m: >
*assignvariableop_63_adam_dense_16_kernel_m:
?
?7
(assignvariableop_64_adam_dense_16_bias_m:	?=
*assignvariableop_65_adam_dense_17_kernel_m:	?6
(assignvariableop_66_adam_dense_17_bias_m:I
+assignvariableop_67_adam_conv3d_15_kernel_v:
 7
)assignvariableop_68_adam_conv3d_15_bias_v: E
7assignvariableop_69_adam_batch_normalization_40_gamma_v: D
6assignvariableop_70_adam_batch_normalization_40_beta_v: I
+assignvariableop_71_adam_conv3d_16_kernel_v:
  7
)assignvariableop_72_adam_conv3d_16_bias_v: E
7assignvariableop_73_adam_batch_normalization_41_gamma_v: D
6assignvariableop_74_adam_batch_normalization_41_beta_v: I
+assignvariableop_75_adam_conv3d_17_kernel_v:
  7
)assignvariableop_76_adam_conv3d_17_bias_v: E
7assignvariableop_77_adam_batch_normalization_42_gamma_v: D
6assignvariableop_78_adam_batch_normalization_42_beta_v: I
+assignvariableop_79_adam_conv3d_18_kernel_v:
  7
)assignvariableop_80_adam_conv3d_18_bias_v: E
7assignvariableop_81_adam_batch_normalization_43_gamma_v: D
6assignvariableop_82_adam_batch_normalization_43_beta_v: I
+assignvariableop_83_adam_conv3d_19_kernel_v:
  7
)assignvariableop_84_adam_conv3d_19_bias_v: E
7assignvariableop_85_adam_batch_normalization_44_gamma_v: D
6assignvariableop_86_adam_batch_normalization_44_beta_v: >
*assignvariableop_87_adam_dense_16_kernel_v:
?
?7
(assignvariableop_88_adam_dense_16_bias_v:	?=
*assignvariableop_89_adam_dense_17_kernel_v:	?6
(assignvariableop_90_adam_dense_17_bias_v:L
.assignvariableop_91_adam_conv3d_15_kernel_vhat:
 :
,assignvariableop_92_adam_conv3d_15_bias_vhat: H
:assignvariableop_93_adam_batch_normalization_40_gamma_vhat: G
9assignvariableop_94_adam_batch_normalization_40_beta_vhat: L
.assignvariableop_95_adam_conv3d_16_kernel_vhat:
  :
,assignvariableop_96_adam_conv3d_16_bias_vhat: H
:assignvariableop_97_adam_batch_normalization_41_gamma_vhat: G
9assignvariableop_98_adam_batch_normalization_41_beta_vhat: L
.assignvariableop_99_adam_conv3d_17_kernel_vhat:
  ;
-assignvariableop_100_adam_conv3d_17_bias_vhat: I
;assignvariableop_101_adam_batch_normalization_42_gamma_vhat: H
:assignvariableop_102_adam_batch_normalization_42_beta_vhat: M
/assignvariableop_103_adam_conv3d_18_kernel_vhat:
  ;
-assignvariableop_104_adam_conv3d_18_bias_vhat: I
;assignvariableop_105_adam_batch_normalization_43_gamma_vhat: H
:assignvariableop_106_adam_batch_normalization_43_beta_vhat: M
/assignvariableop_107_adam_conv3d_19_kernel_vhat:
  ;
-assignvariableop_108_adam_conv3d_19_bias_vhat: I
;assignvariableop_109_adam_batch_normalization_44_gamma_vhat: H
:assignvariableop_110_adam_batch_normalization_44_beta_vhat: B
.assignvariableop_111_adam_dense_16_kernel_vhat:
?
?;
,assignvariableop_112_adam_dense_16_bias_vhat:	?A
.assignvariableop_113_adam_dense_17_kernel_vhat:	?:
,assignvariableop_114_adam_dense_17_bias_vhat:
identity_116??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?C
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?B
value?BB?BtB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?
value?B?tB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypesx
v2t	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv3d_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv3d_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_40_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_40_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_40_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_40_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv3d_16_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv3d_16_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_41_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_41_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_41_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_41_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv3d_17_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv3d_17_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_42_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_42_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_42_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_42_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv3d_18_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv3d_18_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_43_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_43_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_43_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_43_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv3d_19_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv3d_19_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_44_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_44_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_44_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_44_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_16_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp!assignvariableop_31_dense_16_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp#assignvariableop_32_dense_17_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp!assignvariableop_33_dense_17_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_iterIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_beta_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_adam_beta_2Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_decayIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_learning_rateIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_totalIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_countIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv3d_15_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv3d_15_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_batch_normalization_40_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_40_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv3d_16_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv3d_16_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_batch_normalization_41_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_batch_normalization_41_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv3d_17_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv3d_17_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_batch_normalization_42_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_42_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv3d_18_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv3d_18_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_batch_normalization_43_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_batch_normalization_43_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv3d_19_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv3d_19_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp7assignvariableop_61_adam_batch_normalization_44_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp6assignvariableop_62_adam_batch_normalization_44_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_16_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_16_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_17_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_17_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv3d_15_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv3d_15_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_batch_normalization_40_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_batch_normalization_40_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv3d_16_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv3d_16_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp7assignvariableop_73_adam_batch_normalization_41_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp6assignvariableop_74_adam_batch_normalization_41_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv3d_17_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv3d_17_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp7assignvariableop_77_adam_batch_normalization_42_gamma_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adam_batch_normalization_42_beta_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_conv3d_18_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_conv3d_18_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp7assignvariableop_81_adam_batch_normalization_43_gamma_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp6assignvariableop_82_adam_batch_normalization_43_beta_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_conv3d_19_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adam_conv3d_19_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp7assignvariableop_85_adam_batch_normalization_44_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp6assignvariableop_86_adam_batch_normalization_44_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_dense_16_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_dense_16_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_dense_17_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_dense_17_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp.assignvariableop_91_adam_conv3d_15_kernel_vhatIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp,assignvariableop_92_adam_conv3d_15_bias_vhatIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp:assignvariableop_93_adam_batch_normalization_40_gamma_vhatIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp9assignvariableop_94_adam_batch_normalization_40_beta_vhatIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp.assignvariableop_95_adam_conv3d_16_kernel_vhatIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp,assignvariableop_96_adam_conv3d_16_bias_vhatIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp:assignvariableop_97_adam_batch_normalization_41_gamma_vhatIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp9assignvariableop_98_adam_batch_normalization_41_beta_vhatIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp.assignvariableop_99_adam_conv3d_17_kernel_vhatIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp-assignvariableop_100_adam_conv3d_17_bias_vhatIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp;assignvariableop_101_adam_batch_normalization_42_gamma_vhatIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp:assignvariableop_102_adam_batch_normalization_42_beta_vhatIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp/assignvariableop_103_adam_conv3d_18_kernel_vhatIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp-assignvariableop_104_adam_conv3d_18_bias_vhatIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp;assignvariableop_105_adam_batch_normalization_43_gamma_vhatIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp:assignvariableop_106_adam_batch_normalization_43_beta_vhatIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp/assignvariableop_107_adam_conv3d_19_kernel_vhatIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp-assignvariableop_108_adam_conv3d_19_bias_vhatIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp;assignvariableop_109_adam_batch_normalization_44_gamma_vhatIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp:assignvariableop_110_adam_batch_normalization_44_beta_vhatIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp.assignvariableop_111_adam_dense_16_kernel_vhatIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp,assignvariableop_112_adam_dense_16_bias_vhatIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp.assignvariableop_113_adam_dense_17_kernel_vhatIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp,assignvariableop_114_adam_dense_17_bias_vhatIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1149
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_115Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_115?
Identity_116IdentityIdentity_115:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_116"%
identity_116Identity_116:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
7__inference_batch_normalization_41_layer_call_fn_421341

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_4197522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
   : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
   
 
_user_specified_nameinputs
?+
?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421257

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
@@ 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
@@ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
@@ 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
@@ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
@@ 
 
_user_specified_nameinputs
?+
?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_419682

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_419275

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
 :[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
??
?5
__inference__traced_save_422542
file_prefix/
+savev2_conv3d_15_kernel_read_readvariableop-
)savev2_conv3d_15_bias_read_readvariableop;
7savev2_batch_normalization_40_gamma_read_readvariableop:
6savev2_batch_normalization_40_beta_read_readvariableopA
=savev2_batch_normalization_40_moving_mean_read_readvariableopE
Asavev2_batch_normalization_40_moving_variance_read_readvariableop/
+savev2_conv3d_16_kernel_read_readvariableop-
)savev2_conv3d_16_bias_read_readvariableop;
7savev2_batch_normalization_41_gamma_read_readvariableop:
6savev2_batch_normalization_41_beta_read_readvariableopA
=savev2_batch_normalization_41_moving_mean_read_readvariableopE
Asavev2_batch_normalization_41_moving_variance_read_readvariableop/
+savev2_conv3d_17_kernel_read_readvariableop-
)savev2_conv3d_17_bias_read_readvariableop;
7savev2_batch_normalization_42_gamma_read_readvariableop:
6savev2_batch_normalization_42_beta_read_readvariableopA
=savev2_batch_normalization_42_moving_mean_read_readvariableopE
Asavev2_batch_normalization_42_moving_variance_read_readvariableop/
+savev2_conv3d_18_kernel_read_readvariableop-
)savev2_conv3d_18_bias_read_readvariableop;
7savev2_batch_normalization_43_gamma_read_readvariableop:
6savev2_batch_normalization_43_beta_read_readvariableopA
=savev2_batch_normalization_43_moving_mean_read_readvariableopE
Asavev2_batch_normalization_43_moving_variance_read_readvariableop/
+savev2_conv3d_19_kernel_read_readvariableop-
)savev2_conv3d_19_bias_read_readvariableop;
7savev2_batch_normalization_44_gamma_read_readvariableop:
6savev2_batch_normalization_44_beta_read_readvariableopA
=savev2_batch_normalization_44_moving_mean_read_readvariableopE
Asavev2_batch_normalization_44_moving_variance_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv3d_15_kernel_m_read_readvariableop4
0savev2_adam_conv3d_15_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_40_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_40_beta_m_read_readvariableop6
2savev2_adam_conv3d_16_kernel_m_read_readvariableop4
0savev2_adam_conv3d_16_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_41_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_41_beta_m_read_readvariableop6
2savev2_adam_conv3d_17_kernel_m_read_readvariableop4
0savev2_adam_conv3d_17_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_42_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_42_beta_m_read_readvariableop6
2savev2_adam_conv3d_18_kernel_m_read_readvariableop4
0savev2_adam_conv3d_18_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_43_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_43_beta_m_read_readvariableop6
2savev2_adam_conv3d_19_kernel_m_read_readvariableop4
0savev2_adam_conv3d_19_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_44_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_44_beta_m_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop3
/savev2_adam_dense_16_bias_m_read_readvariableop5
1savev2_adam_dense_17_kernel_m_read_readvariableop3
/savev2_adam_dense_17_bias_m_read_readvariableop6
2savev2_adam_conv3d_15_kernel_v_read_readvariableop4
0savev2_adam_conv3d_15_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_40_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_40_beta_v_read_readvariableop6
2savev2_adam_conv3d_16_kernel_v_read_readvariableop4
0savev2_adam_conv3d_16_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_41_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_41_beta_v_read_readvariableop6
2savev2_adam_conv3d_17_kernel_v_read_readvariableop4
0savev2_adam_conv3d_17_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_42_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_42_beta_v_read_readvariableop6
2savev2_adam_conv3d_18_kernel_v_read_readvariableop4
0savev2_adam_conv3d_18_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_43_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_43_beta_v_read_readvariableop6
2savev2_adam_conv3d_19_kernel_v_read_readvariableop4
0savev2_adam_conv3d_19_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_44_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_44_beta_v_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop3
/savev2_adam_dense_16_bias_v_read_readvariableop5
1savev2_adam_dense_17_kernel_v_read_readvariableop3
/savev2_adam_dense_17_bias_v_read_readvariableop9
5savev2_adam_conv3d_15_kernel_vhat_read_readvariableop7
3savev2_adam_conv3d_15_bias_vhat_read_readvariableopE
Asavev2_adam_batch_normalization_40_gamma_vhat_read_readvariableopD
@savev2_adam_batch_normalization_40_beta_vhat_read_readvariableop9
5savev2_adam_conv3d_16_kernel_vhat_read_readvariableop7
3savev2_adam_conv3d_16_bias_vhat_read_readvariableopE
Asavev2_adam_batch_normalization_41_gamma_vhat_read_readvariableopD
@savev2_adam_batch_normalization_41_beta_vhat_read_readvariableop9
5savev2_adam_conv3d_17_kernel_vhat_read_readvariableop7
3savev2_adam_conv3d_17_bias_vhat_read_readvariableopE
Asavev2_adam_batch_normalization_42_gamma_vhat_read_readvariableopD
@savev2_adam_batch_normalization_42_beta_vhat_read_readvariableop9
5savev2_adam_conv3d_18_kernel_vhat_read_readvariableop7
3savev2_adam_conv3d_18_bias_vhat_read_readvariableopE
Asavev2_adam_batch_normalization_43_gamma_vhat_read_readvariableopD
@savev2_adam_batch_normalization_43_beta_vhat_read_readvariableop9
5savev2_adam_conv3d_19_kernel_vhat_read_readvariableop7
3savev2_adam_conv3d_19_bias_vhat_read_readvariableopE
Asavev2_adam_batch_normalization_44_gamma_vhat_read_readvariableopD
@savev2_adam_batch_normalization_44_beta_vhat_read_readvariableop8
4savev2_adam_dense_16_kernel_vhat_read_readvariableop6
2savev2_adam_dense_16_bias_vhat_read_readvariableop8
4savev2_adam_dense_17_kernel_vhat_read_readvariableop6
2savev2_adam_dense_17_bias_vhat_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?C
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?B
value?BB?BtB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?
value?B?tB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?3
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv3d_15_kernel_read_readvariableop)savev2_conv3d_15_bias_read_readvariableop7savev2_batch_normalization_40_gamma_read_readvariableop6savev2_batch_normalization_40_beta_read_readvariableop=savev2_batch_normalization_40_moving_mean_read_readvariableopAsavev2_batch_normalization_40_moving_variance_read_readvariableop+savev2_conv3d_16_kernel_read_readvariableop)savev2_conv3d_16_bias_read_readvariableop7savev2_batch_normalization_41_gamma_read_readvariableop6savev2_batch_normalization_41_beta_read_readvariableop=savev2_batch_normalization_41_moving_mean_read_readvariableopAsavev2_batch_normalization_41_moving_variance_read_readvariableop+savev2_conv3d_17_kernel_read_readvariableop)savev2_conv3d_17_bias_read_readvariableop7savev2_batch_normalization_42_gamma_read_readvariableop6savev2_batch_normalization_42_beta_read_readvariableop=savev2_batch_normalization_42_moving_mean_read_readvariableopAsavev2_batch_normalization_42_moving_variance_read_readvariableop+savev2_conv3d_18_kernel_read_readvariableop)savev2_conv3d_18_bias_read_readvariableop7savev2_batch_normalization_43_gamma_read_readvariableop6savev2_batch_normalization_43_beta_read_readvariableop=savev2_batch_normalization_43_moving_mean_read_readvariableopAsavev2_batch_normalization_43_moving_variance_read_readvariableop+savev2_conv3d_19_kernel_read_readvariableop)savev2_conv3d_19_bias_read_readvariableop7savev2_batch_normalization_44_gamma_read_readvariableop6savev2_batch_normalization_44_beta_read_readvariableop=savev2_batch_normalization_44_moving_mean_read_readvariableopAsavev2_batch_normalization_44_moving_variance_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv3d_15_kernel_m_read_readvariableop0savev2_adam_conv3d_15_bias_m_read_readvariableop>savev2_adam_batch_normalization_40_gamma_m_read_readvariableop=savev2_adam_batch_normalization_40_beta_m_read_readvariableop2savev2_adam_conv3d_16_kernel_m_read_readvariableop0savev2_adam_conv3d_16_bias_m_read_readvariableop>savev2_adam_batch_normalization_41_gamma_m_read_readvariableop=savev2_adam_batch_normalization_41_beta_m_read_readvariableop2savev2_adam_conv3d_17_kernel_m_read_readvariableop0savev2_adam_conv3d_17_bias_m_read_readvariableop>savev2_adam_batch_normalization_42_gamma_m_read_readvariableop=savev2_adam_batch_normalization_42_beta_m_read_readvariableop2savev2_adam_conv3d_18_kernel_m_read_readvariableop0savev2_adam_conv3d_18_bias_m_read_readvariableop>savev2_adam_batch_normalization_43_gamma_m_read_readvariableop=savev2_adam_batch_normalization_43_beta_m_read_readvariableop2savev2_adam_conv3d_19_kernel_m_read_readvariableop0savev2_adam_conv3d_19_bias_m_read_readvariableop>savev2_adam_batch_normalization_44_gamma_m_read_readvariableop=savev2_adam_batch_normalization_44_beta_m_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop/savev2_adam_dense_16_bias_m_read_readvariableop1savev2_adam_dense_17_kernel_m_read_readvariableop/savev2_adam_dense_17_bias_m_read_readvariableop2savev2_adam_conv3d_15_kernel_v_read_readvariableop0savev2_adam_conv3d_15_bias_v_read_readvariableop>savev2_adam_batch_normalization_40_gamma_v_read_readvariableop=savev2_adam_batch_normalization_40_beta_v_read_readvariableop2savev2_adam_conv3d_16_kernel_v_read_readvariableop0savev2_adam_conv3d_16_bias_v_read_readvariableop>savev2_adam_batch_normalization_41_gamma_v_read_readvariableop=savev2_adam_batch_normalization_41_beta_v_read_readvariableop2savev2_adam_conv3d_17_kernel_v_read_readvariableop0savev2_adam_conv3d_17_bias_v_read_readvariableop>savev2_adam_batch_normalization_42_gamma_v_read_readvariableop=savev2_adam_batch_normalization_42_beta_v_read_readvariableop2savev2_adam_conv3d_18_kernel_v_read_readvariableop0savev2_adam_conv3d_18_bias_v_read_readvariableop>savev2_adam_batch_normalization_43_gamma_v_read_readvariableop=savev2_adam_batch_normalization_43_beta_v_read_readvariableop2savev2_adam_conv3d_19_kernel_v_read_readvariableop0savev2_adam_conv3d_19_bias_v_read_readvariableop>savev2_adam_batch_normalization_44_gamma_v_read_readvariableop=savev2_adam_batch_normalization_44_beta_v_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop/savev2_adam_dense_16_bias_v_read_readvariableop1savev2_adam_dense_17_kernel_v_read_readvariableop/savev2_adam_dense_17_bias_v_read_readvariableop5savev2_adam_conv3d_15_kernel_vhat_read_readvariableop3savev2_adam_conv3d_15_bias_vhat_read_readvariableopAsavev2_adam_batch_normalization_40_gamma_vhat_read_readvariableop@savev2_adam_batch_normalization_40_beta_vhat_read_readvariableop5savev2_adam_conv3d_16_kernel_vhat_read_readvariableop3savev2_adam_conv3d_16_bias_vhat_read_readvariableopAsavev2_adam_batch_normalization_41_gamma_vhat_read_readvariableop@savev2_adam_batch_normalization_41_beta_vhat_read_readvariableop5savev2_adam_conv3d_17_kernel_vhat_read_readvariableop3savev2_adam_conv3d_17_bias_vhat_read_readvariableopAsavev2_adam_batch_normalization_42_gamma_vhat_read_readvariableop@savev2_adam_batch_normalization_42_beta_vhat_read_readvariableop5savev2_adam_conv3d_18_kernel_vhat_read_readvariableop3savev2_adam_conv3d_18_bias_vhat_read_readvariableopAsavev2_adam_batch_normalization_43_gamma_vhat_read_readvariableop@savev2_adam_batch_normalization_43_beta_vhat_read_readvariableop5savev2_adam_conv3d_19_kernel_vhat_read_readvariableop3savev2_adam_conv3d_19_bias_vhat_read_readvariableopAsavev2_adam_batch_normalization_44_gamma_vhat_read_readvariableop@savev2_adam_batch_normalization_44_beta_vhat_read_readvariableop4savev2_adam_dense_16_kernel_vhat_read_readvariableop2savev2_adam_dense_16_bias_vhat_read_readvariableop4savev2_adam_dense_17_kernel_vhat_read_readvariableop2savev2_adam_dense_17_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypesx
v2t	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
 : : : : : :
  : : : : : :
  : : : : : :
  : : : : : :
  : : : : : :
?
?:?:	?:: : : : : : : : : :
 : : : :
  : : : :
  : : : :
  : : : :
  : : : :
?
?:?:	?::
 : : : :
  : : : :
  : : : :
  : : : :
  : : : :
?
?:?:	?::
 : : : :
  : : : :
  : : : :
  : : : :
  : : : :
?
?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:
 : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
:
  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
:
  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
:
  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
:
  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
?
?:! 

_output_shapes	
:?:%!!

_output_shapes
:	?: "

_output_shapes
::#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :0,,
*
_output_shapes
:
 : -

_output_shapes
: : .

_output_shapes
: : /

_output_shapes
: :00,
*
_output_shapes
:
  : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: :04,
*
_output_shapes
:
  : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :08,
*
_output_shapes
:
  : 9

_output_shapes
: : :

_output_shapes
: : ;

_output_shapes
: :0<,
*
_output_shapes
:
  : =

_output_shapes
: : >

_output_shapes
: : ?

_output_shapes
: :&@"
 
_output_shapes
:
?
?:!A

_output_shapes	
:?:%B!

_output_shapes
:	?: C

_output_shapes
::0D,
*
_output_shapes
:
 : E

_output_shapes
: : F

_output_shapes
: : G

_output_shapes
: :0H,
*
_output_shapes
:
  : I

_output_shapes
: : J

_output_shapes
: : K

_output_shapes
: :0L,
*
_output_shapes
:
  : M

_output_shapes
: : N

_output_shapes
: : O

_output_shapes
: :0P,
*
_output_shapes
:
  : Q

_output_shapes
: : R

_output_shapes
: : S

_output_shapes
: :0T,
*
_output_shapes
:
  : U

_output_shapes
: : V

_output_shapes
: : W

_output_shapes
: :&X"
 
_output_shapes
:
?
?:!Y

_output_shapes	
:?:%Z!

_output_shapes
:	?: [

_output_shapes
::0\,
*
_output_shapes
:
 : ]

_output_shapes
: : ^

_output_shapes
: : _

_output_shapes
: :0`,
*
_output_shapes
:
  : a

_output_shapes
: : b

_output_shapes
: : c

_output_shapes
: :0d,
*
_output_shapes
:
  : e

_output_shapes
: : f

_output_shapes
: : g

_output_shapes
: :0h,
*
_output_shapes
:
  : i

_output_shapes
: : j

_output_shapes
: : k

_output_shapes
: :0l,
*
_output_shapes
:
  : m

_output_shapes
: : n

_output_shapes
: : o

_output_shapes
: :&p"
 
_output_shapes
:
?
?:!q

_output_shapes	
:?:%r!

_output_shapes
:	?: s

_output_shapes
::t

_output_shapes
: 
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_419205

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_420294
conv3d_15_input.
conv3d_15_420181:
 
conv3d_15_420183: +
batch_normalization_40_420186: +
batch_normalization_40_420188: +
batch_normalization_40_420190: +
batch_normalization_40_420192: .
conv3d_16_420196:
  
conv3d_16_420198: +
batch_normalization_41_420201: +
batch_normalization_41_420203: +
batch_normalization_41_420205: +
batch_normalization_41_420207: .
conv3d_17_420211:
  
conv3d_17_420213: +
batch_normalization_42_420216: +
batch_normalization_42_420218: +
batch_normalization_42_420220: +
batch_normalization_42_420222: .
conv3d_18_420226:
  
conv3d_18_420228: +
batch_normalization_43_420231: +
batch_normalization_43_420233: +
batch_normalization_43_420235: +
batch_normalization_43_420237: .
conv3d_19_420241:
  
conv3d_19_420243: +
batch_normalization_44_420246: +
batch_normalization_44_420248: +
batch_normalization_44_420250: +
batch_normalization_44_420252: #
dense_16_420258:
?
?
dense_16_420260:	?"
dense_17_420264:	?
dense_17_420266:
identity??.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?!conv3d_15/StatefulPartitionedCall?!conv3d_16/StatefulPartitionedCall?2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?!conv3d_17/StatefulPartitionedCall?2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?!conv3d_18/StatefulPartitionedCall?2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?!conv3d_19/StatefulPartitionedCall?2conv3d_19/kernel/Regularizer/Square/ReadVariableOp? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCallconv3d_15_inputconv3d_15_420181conv3d_15_420183*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_4190212#
!conv3d_15/StatefulPartitionedCall?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0batch_normalization_40_420186batch_normalization_40_420188batch_normalization_40_420190batch_normalization_40_420192*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_41904620
.batch_normalization_40/StatefulPartitionedCall?
 max_pooling3d_30/PartitionedCallPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_30_layer_call_and_return_conditional_losses_4183012"
 max_pooling3d_30/PartitionedCall?
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_30/PartitionedCall:output:0conv3d_16_420196conv3d_16_420198*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_4190742#
!conv3d_16/StatefulPartitionedCall?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0batch_normalization_41_420201batch_normalization_41_420203batch_normalization_41_420205batch_normalization_41_420207*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_41909920
.batch_normalization_41/StatefulPartitionedCall?
 max_pooling3d_31/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_31_layer_call_and_return_conditional_losses_4184752"
 max_pooling3d_31/PartitionedCall?
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_31/PartitionedCall:output:0conv3d_17_420211conv3d_17_420213*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_4191272#
!conv3d_17/StatefulPartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0batch_normalization_42_420216batch_normalization_42_420218batch_normalization_42_420220batch_normalization_42_420222*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_41915220
.batch_normalization_42/StatefulPartitionedCall?
 max_pooling3d_32/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_32_layer_call_and_return_conditional_losses_4186492"
 max_pooling3d_32/PartitionedCall?
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_32/PartitionedCall:output:0conv3d_18_420226conv3d_18_420228*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_18_layer_call_and_return_conditional_losses_4191802#
!conv3d_18/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0batch_normalization_43_420231batch_normalization_43_420233batch_normalization_43_420235batch_normalization_43_420237*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_41920520
.batch_normalization_43/StatefulPartitionedCall?
 max_pooling3d_33/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_33_layer_call_and_return_conditional_losses_4188232"
 max_pooling3d_33/PartitionedCall?
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_33/PartitionedCall:output:0conv3d_19_420241conv3d_19_420243*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_19_layer_call_and_return_conditional_losses_4192332#
!conv3d_19/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0batch_normalization_44_420246batch_normalization_44_420248batch_normalization_44_420250batch_normalization_44_420252*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_41925820
.batch_normalization_44/StatefulPartitionedCall?
 max_pooling3d_34/PartitionedCallPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_34_layer_call_and_return_conditional_losses_4189972"
 max_pooling3d_34/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall)max_pooling3d_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_4192752
flatten_8/PartitionedCall?
dropout_16/PartitionedCallPartitionedCall"flatten_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_4192822
dropout_16/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_16_420258dense_16_420260*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_4192952"
 dense_16/StatefulPartitionedCall?
dropout_17/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_4193062
dropout_17/PartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_17_420264dense_17_420266*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_4193192"
 dense_17/StatefulPartitionedCall?
2conv3d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_16_420196**
_output_shapes
:
  *
dtype024
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_16/kernel/Regularizer/SquareSquare:conv3d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_16/kernel/Regularizer/Square?
"conv3d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_16/kernel/Regularizer/Const?
 conv3d_16/kernel/Regularizer/SumSum'conv3d_16/kernel/Regularizer/Square:y:0+conv3d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/Sum?
"conv3d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_16/kernel/Regularizer/mul/x?
 conv3d_16/kernel/Regularizer/mulMul+conv3d_16/kernel/Regularizer/mul/x:output:0)conv3d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/mul?
2conv3d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_17_420211**
_output_shapes
:
  *
dtype024
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_17/kernel/Regularizer/SquareSquare:conv3d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_17/kernel/Regularizer/Square?
"conv3d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_17/kernel/Regularizer/Const?
 conv3d_17/kernel/Regularizer/SumSum'conv3d_17/kernel/Regularizer/Square:y:0+conv3d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/Sum?
"conv3d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_17/kernel/Regularizer/mul/x?
 conv3d_17/kernel/Regularizer/mulMul+conv3d_17/kernel/Regularizer/mul/x:output:0)conv3d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/mul?
2conv3d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_18_420226**
_output_shapes
:
  *
dtype024
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_18/kernel/Regularizer/SquareSquare:conv3d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_18/kernel/Regularizer/Square?
"conv3d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_18/kernel/Regularizer/Const?
 conv3d_18/kernel/Regularizer/SumSum'conv3d_18/kernel/Regularizer/Square:y:0+conv3d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/Sum?
"conv3d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_18/kernel/Regularizer/mul/x?
 conv3d_18/kernel/Regularizer/mulMul+conv3d_18/kernel/Regularizer/mul/x:output:0)conv3d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/mul?
2conv3d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_19_420241**
_output_shapes
:
  *
dtype024
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_19/kernel/Regularizer/SquareSquare:conv3d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_19/kernel/Regularizer/Square?
"conv3d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_19/kernel/Regularizer/Const?
 conv3d_19/kernel/Regularizer/SumSum'conv3d_19/kernel/Regularizer/Square:y:0+conv3d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/Sum?
"conv3d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_19/kernel/Regularizer/mul/x?
 conv3d_19/kernel/Regularizer/mulMul+conv3d_19/kernel/Regularizer/mul/x:output:0)conv3d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/mul?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall3^conv3d_16/kernel/Regularizer/Square/ReadVariableOp"^conv3d_17/StatefulPartitionedCall3^conv3d_17/kernel/Regularizer/Square/ReadVariableOp"^conv3d_18/StatefulPartitionedCall3^conv3d_18/kernel/Regularizer/Square/ReadVariableOp"^conv3d_19/StatefulPartitionedCall3^conv3d_19/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2h
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2h
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_18/StatefulPartitionedCall!conv3d_18/StatefulPartitionedCall2h
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_19/StatefulPartitionedCall!conv3d_19/StatefulPartitionedCall2h
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:d `
3
_output_shapes!
:?????????
@@
)
_user_specified_nameconv3d_15_input
?
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_419306

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_422141Y
;conv3d_16_kernel_regularizer_square_readvariableop_resource:
  
identity??2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
2conv3d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv3d_16_kernel_regularizer_square_readvariableop_resource**
_output_shapes
:
  *
dtype024
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_16/kernel/Regularizer/SquareSquare:conv3d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_16/kernel/Regularizer/Square?
"conv3d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_16/kernel/Regularizer/Const?
 conv3d_16/kernel/Regularizer/SumSum'conv3d_16/kernel/Regularizer/Square:y:0+conv3d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/Sum?
"conv3d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_16/kernel/Regularizer/mul/x?
 conv3d_16/kernel/Regularizer/mulMul+conv3d_16/kernel/Regularizer/mul/x:output:0)conv3d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/mul?
IdentityIdentity$conv3d_16/kernel/Regularizer/mul:z:03^conv3d_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2conv3d_16/kernel/Regularizer/Square/ReadVariableOp
?
h
L__inference_max_pooling3d_33_layer_call_and_return_conditional_losses_418823

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?+
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_422025

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_420410
conv3d_15_input.
conv3d_15_420297:
 
conv3d_15_420299: +
batch_normalization_40_420302: +
batch_normalization_40_420304: +
batch_normalization_40_420306: +
batch_normalization_40_420308: .
conv3d_16_420312:
  
conv3d_16_420314: +
batch_normalization_41_420317: +
batch_normalization_41_420319: +
batch_normalization_41_420321: +
batch_normalization_41_420323: .
conv3d_17_420327:
  
conv3d_17_420329: +
batch_normalization_42_420332: +
batch_normalization_42_420334: +
batch_normalization_42_420336: +
batch_normalization_42_420338: .
conv3d_18_420342:
  
conv3d_18_420344: +
batch_normalization_43_420347: +
batch_normalization_43_420349: +
batch_normalization_43_420351: +
batch_normalization_43_420353: .
conv3d_19_420357:
  
conv3d_19_420359: +
batch_normalization_44_420362: +
batch_normalization_44_420364: +
batch_normalization_44_420366: +
batch_normalization_44_420368: #
dense_16_420374:
?
?
dense_16_420376:	?"
dense_17_420380:	?
dense_17_420382:
identity??.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?!conv3d_15/StatefulPartitionedCall?!conv3d_16/StatefulPartitionedCall?2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?!conv3d_17/StatefulPartitionedCall?2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?!conv3d_18/StatefulPartitionedCall?2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?!conv3d_19/StatefulPartitionedCall?2conv3d_19/kernel/Regularizer/Square/ReadVariableOp? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?"dropout_17/StatefulPartitionedCall?
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCallconv3d_15_inputconv3d_15_420297conv3d_15_420299*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_4190212#
!conv3d_15/StatefulPartitionedCall?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0batch_normalization_40_420302batch_normalization_40_420304batch_normalization_40_420306batch_normalization_40_420308*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_41982220
.batch_normalization_40/StatefulPartitionedCall?
 max_pooling3d_30/PartitionedCallPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_30_layer_call_and_return_conditional_losses_4183012"
 max_pooling3d_30/PartitionedCall?
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_30/PartitionedCall:output:0conv3d_16_420312conv3d_16_420314*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_4190742#
!conv3d_16/StatefulPartitionedCall?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0batch_normalization_41_420317batch_normalization_41_420319batch_normalization_41_420321batch_normalization_41_420323*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_41975220
.batch_normalization_41/StatefulPartitionedCall?
 max_pooling3d_31/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_31_layer_call_and_return_conditional_losses_4184752"
 max_pooling3d_31/PartitionedCall?
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_31/PartitionedCall:output:0conv3d_17_420327conv3d_17_420329*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_4191272#
!conv3d_17/StatefulPartitionedCall?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0batch_normalization_42_420332batch_normalization_42_420334batch_normalization_42_420336batch_normalization_42_420338*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_41968220
.batch_normalization_42/StatefulPartitionedCall?
 max_pooling3d_32/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_32_layer_call_and_return_conditional_losses_4186492"
 max_pooling3d_32/PartitionedCall?
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_32/PartitionedCall:output:0conv3d_18_420342conv3d_18_420344*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_18_layer_call_and_return_conditional_losses_4191802#
!conv3d_18/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv3d_18/StatefulPartitionedCall:output:0batch_normalization_43_420347batch_normalization_43_420349batch_normalization_43_420351batch_normalization_43_420353*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_41961220
.batch_normalization_43/StatefulPartitionedCall?
 max_pooling3d_33/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_33_layer_call_and_return_conditional_losses_4188232"
 max_pooling3d_33/PartitionedCall?
!conv3d_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_33/PartitionedCall:output:0conv3d_19_420357conv3d_19_420359*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_19_layer_call_and_return_conditional_losses_4192332#
!conv3d_19/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv3d_19/StatefulPartitionedCall:output:0batch_normalization_44_420362batch_normalization_44_420364batch_normalization_44_420366batch_normalization_44_420368*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_41954220
.batch_normalization_44/StatefulPartitionedCall?
 max_pooling3d_34/PartitionedCallPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_34_layer_call_and_return_conditional_losses_4189972"
 max_pooling3d_34/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall)max_pooling3d_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_4192752
flatten_8/PartitionedCall?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_4194842$
"dropout_16/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_16_420374dense_16_420376*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_4192952"
 dense_16/StatefulPartitionedCall?
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_4194512$
"dropout_17/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_17_420380dense_17_420382*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_4193192"
 dense_17/StatefulPartitionedCall?
2conv3d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_16_420312**
_output_shapes
:
  *
dtype024
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_16/kernel/Regularizer/SquareSquare:conv3d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_16/kernel/Regularizer/Square?
"conv3d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_16/kernel/Regularizer/Const?
 conv3d_16/kernel/Regularizer/SumSum'conv3d_16/kernel/Regularizer/Square:y:0+conv3d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/Sum?
"conv3d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_16/kernel/Regularizer/mul/x?
 conv3d_16/kernel/Regularizer/mulMul+conv3d_16/kernel/Regularizer/mul/x:output:0)conv3d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_16/kernel/Regularizer/mul?
2conv3d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_17_420327**
_output_shapes
:
  *
dtype024
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_17/kernel/Regularizer/SquareSquare:conv3d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_17/kernel/Regularizer/Square?
"conv3d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_17/kernel/Regularizer/Const?
 conv3d_17/kernel/Regularizer/SumSum'conv3d_17/kernel/Regularizer/Square:y:0+conv3d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/Sum?
"conv3d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_17/kernel/Regularizer/mul/x?
 conv3d_17/kernel/Regularizer/mulMul+conv3d_17/kernel/Regularizer/mul/x:output:0)conv3d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_17/kernel/Regularizer/mul?
2conv3d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_18_420342**
_output_shapes
:
  *
dtype024
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_18/kernel/Regularizer/SquareSquare:conv3d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_18/kernel/Regularizer/Square?
"conv3d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_18/kernel/Regularizer/Const?
 conv3d_18/kernel/Regularizer/SumSum'conv3d_18/kernel/Regularizer/Square:y:0+conv3d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/Sum?
"conv3d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_18/kernel/Regularizer/mul/x?
 conv3d_18/kernel/Regularizer/mulMul+conv3d_18/kernel/Regularizer/mul/x:output:0)conv3d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_18/kernel/Regularizer/mul?
2conv3d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_19_420357**
_output_shapes
:
  *
dtype024
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp?
#conv3d_19/kernel/Regularizer/SquareSquare:conv3d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:
  2%
#conv3d_19/kernel/Regularizer/Square?
"conv3d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                2$
"conv3d_19/kernel/Regularizer/Const?
 conv3d_19/kernel/Regularizer/SumSum'conv3d_19/kernel/Regularizer/Square:y:0+conv3d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/Sum?
"conv3d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;2$
"conv3d_19/kernel/Regularizer/mul/x?
 conv3d_19/kernel/Regularizer/mulMul+conv3d_19/kernel/Regularizer/mul/x:output:0)conv3d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv3d_19/kernel/Regularizer/mul?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall3^conv3d_16/kernel/Regularizer/Square/ReadVariableOp"^conv3d_17/StatefulPartitionedCall3^conv3d_17/kernel/Regularizer/Square/ReadVariableOp"^conv3d_18/StatefulPartitionedCall3^conv3d_18/kernel/Regularizer/Square/ReadVariableOp"^conv3d_19/StatefulPartitionedCall3^conv3d_19/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????
@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2h
2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2conv3d_16/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2h
2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2conv3d_17/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_18/StatefulPartitionedCall!conv3d_18/StatefulPartitionedCall2h
2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2conv3d_18/kernel/Regularizer/Square/ReadVariableOp2F
!conv3d_19/StatefulPartitionedCall!conv3d_19/StatefulPartitionedCall2h
2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2conv3d_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall:d `
3
_output_shapes!
:?????????
@@
)
_user_specified_nameconv3d_15_input
?	
?
7__inference_batch_normalization_41_layer_call_fn_421302

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_4183312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_34_layer_call_and_return_conditional_losses_418997

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_42_layer_call_fn_421533

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_4196822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?,
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_418739

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?+
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421833

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:?????????
 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_421991

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:?????????
 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:?????????
 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
 : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_31_layer_call_and_return_conditional_losses_418475

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
W
conv3d_15_inputD
!serving_default_conv3d_15_input:0?????????
@@<
dense_170
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer-15
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_sequential??{"name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 64, 64, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv3d_15_input"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 64, 64, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_40", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_30", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_41", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_31", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_17", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_42", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_32", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_19", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 59, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 1}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 64, 64, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 10, 64, 64, 1]}, "float32", "conv3d_15_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 64, 64, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv3d_15_input"}, "shared_object_id": 0}, {"class_name": "Conv3D", "config": {"name": "conv3d_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 64, 64, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_40", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_30", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 9}, {"class_name": "Conv3D", "config": {"name": "conv3d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}, "shared_object_id": 12}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_41", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 18}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_31", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 19}, {"class_name": "Conv3D", "config": {"name": "conv3d_17", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}, "shared_object_id": 22}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_42", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 28}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_32", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 29}, {"class_name": "Conv3D", "config": {"name": "conv3d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}, "shared_object_id": 32}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 33}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 35}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 37}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 38}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 39}, {"class_name": "Conv3D", "config": {"name": "conv3d_19", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}, "shared_object_id": 42}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 43}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 45}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 47}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 48}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 49}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 50}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "shared_object_id": 51}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 54}, {"class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "shared_object_id": 55}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 56}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 58}]}}, "training_config": {"loss": "kullback_leibler_divergence", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 61}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": true}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "conv3d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 64, 64, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 64, 64, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 1}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 64, 64, 1]}}
?

!axis
	"gamma
#beta
$moving_mean
%moving_variance
&	variables
'regularization_losses
(trainable_variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_40", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_40", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 32}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 64, 64, 32]}}
?
*	variables
+regularization_losses
,trainable_variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling3d_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_30", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 63}}
?

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "conv3d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}, "shared_object_id": 12}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 32}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 32, 32, 32]}}
?

4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:regularization_losses
;trainable_variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_41", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_41", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 32}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 32, 32, 32]}}
?
=	variables
>regularization_losses
?trainable_variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling3d_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_31", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 66}}
?

Akernel
Bbias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "conv3d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_17", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}, "shared_object_id": 22}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 32}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 16, 16, 32]}}
?

Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_42", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_42", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 32}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 16, 16, 32]}}
?
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling3d_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_32", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 69}}
?

Tkernel
Ubias
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "conv3d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}, "shared_object_id": 32}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 32}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 8, 8, 32]}}
?

Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_	variables
`regularization_losses
atrainable_variables
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_43", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 35}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 37}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 32}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 8, 8, 32]}}
?
c	variables
dregularization_losses
etrainable_variables
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling3d_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 72}}
?

gkernel
hbias
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "conv3d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_19", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [10, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.004999999888241291}, "shared_object_id": 42}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 32}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 4, 4, 32]}}
?

maxis
	ngamma
obeta
pmoving_mean
qmoving_variance
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_44", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 45}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 47}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 48, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 32}}, "shared_object_id": 74}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 4, 4, 32]}}
?
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling3d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 49, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 75}}
?
z	variables
{regularization_losses
|trainable_variables
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 50, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 76}}
?
~	variables
regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "shared_object_id": 51}
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1280}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1280]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "shared_object_id": 55}
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 56}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 58, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 78}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?"m?#m?.m?/m?5m?6m?Am?Bm?Hm?Im?Tm?Um?[m?\m?gm?hm?nm?om?	?m?	?m?	?m?	?m?v?v?"v?#v?.v?/v?5v?6v?Av?Bv?Hv?Iv?Tv?Uv?[v?\v?gv?hv?nv?ov?	?v?	?v?	?v?	?v?vhat?vhat?"vhat?#vhat?.vhat?/vhat?5vhat?6vhat?Avhat?Bvhat?Hvhat?Ivhat?Tvhat?Uvhat?[vhat?\vhat?gvhat?hvhat?nvhat?ovhat??vhat??vhat??vhat??vhat?"
	optimizer
?
0
1
"2
#3
$4
%5
.6
/7
58
69
710
811
A12
B13
H14
I15
J16
K17
T18
U19
[20
\21
]22
^23
g24
h25
n26
o27
p28
q29
?30
?31
?32
?33"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
0
1
"2
#3
.4
/5
56
67
A8
B9
H10
I11
T12
U13
[14
\15
g16
h17
n18
o19
?20
?21
?22
?23"
trackable_list_wrapper
?
?non_trainable_variables
	variables
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
.:,
 2conv3d_15/kernel
: 2conv3d_15/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?non_trainable_variables
	variables
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_40/gamma
):' 2batch_normalization_40/beta
2:0  (2"batch_normalization_40/moving_mean
6:4  (2&batch_normalization_40/moving_variance
<
"0
#1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
?non_trainable_variables
&	variables
'regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
(trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
*	variables
+regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,
  2conv3d_16/kernel
: 2conv3d_16/bias
.
.0
/1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
?non_trainable_variables
0	variables
1regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
2trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_41/gamma
):' 2batch_normalization_41/beta
2:0  (2"batch_normalization_41/moving_mean
6:4  (2&batch_normalization_41/moving_variance
<
50
61
72
83"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
?non_trainable_variables
9	variables
:regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
;trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
=	variables
>regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,
  2conv3d_17/kernel
: 2conv3d_17/bias
.
A0
B1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
?non_trainable_variables
C	variables
Dregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
Etrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_42/gamma
):' 2batch_normalization_42/beta
2:0  (2"batch_normalization_42/moving_mean
6:4  (2&batch_normalization_42/moving_variance
<
H0
I1
J2
K3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
?non_trainable_variables
L	variables
Mregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
Ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
P	variables
Qregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
Rtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,
  2conv3d_18/kernel
: 2conv3d_18/bias
.
T0
U1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
?
?non_trainable_variables
V	variables
Wregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
Xtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_43/gamma
):' 2batch_normalization_43/beta
2:0  (2"batch_normalization_43/moving_mean
6:4  (2&batch_normalization_43/moving_variance
<
[0
\1
]2
^3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
?
?non_trainable_variables
_	variables
`regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
atrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
c	variables
dregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
etrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,
  2conv3d_19/kernel
: 2conv3d_19/bias
.
g0
h1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
?
?non_trainable_variables
i	variables
jregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
ktrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_44/gamma
):' 2batch_normalization_44/beta
2:0  (2"batch_normalization_44/moving_mean
6:4  (2&batch_normalization_44/moving_variance
<
n0
o1
p2
q3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
?
?non_trainable_variables
r	variables
sregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
ttrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
v	variables
wregularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
xtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
z	variables
{regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
|trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
~	variables
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
?
?2dense_16/kernel
:?2dense_16/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_17/kernel
:2dense_17/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
f
$0
%1
72
83
J4
K5
]6
^7
p8
q9"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 79}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 61}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
3:1
 2Adam/conv3d_15/kernel/m
!: 2Adam/conv3d_15/bias/m
/:- 2#Adam/batch_normalization_40/gamma/m
.:, 2"Adam/batch_normalization_40/beta/m
3:1
  2Adam/conv3d_16/kernel/m
!: 2Adam/conv3d_16/bias/m
/:- 2#Adam/batch_normalization_41/gamma/m
.:, 2"Adam/batch_normalization_41/beta/m
3:1
  2Adam/conv3d_17/kernel/m
!: 2Adam/conv3d_17/bias/m
/:- 2#Adam/batch_normalization_42/gamma/m
.:, 2"Adam/batch_normalization_42/beta/m
3:1
  2Adam/conv3d_18/kernel/m
!: 2Adam/conv3d_18/bias/m
/:- 2#Adam/batch_normalization_43/gamma/m
.:, 2"Adam/batch_normalization_43/beta/m
3:1
  2Adam/conv3d_19/kernel/m
!: 2Adam/conv3d_19/bias/m
/:- 2#Adam/batch_normalization_44/gamma/m
.:, 2"Adam/batch_normalization_44/beta/m
(:&
?
?2Adam/dense_16/kernel/m
!:?2Adam/dense_16/bias/m
':%	?2Adam/dense_17/kernel/m
 :2Adam/dense_17/bias/m
3:1
 2Adam/conv3d_15/kernel/v
!: 2Adam/conv3d_15/bias/v
/:- 2#Adam/batch_normalization_40/gamma/v
.:, 2"Adam/batch_normalization_40/beta/v
3:1
  2Adam/conv3d_16/kernel/v
!: 2Adam/conv3d_16/bias/v
/:- 2#Adam/batch_normalization_41/gamma/v
.:, 2"Adam/batch_normalization_41/beta/v
3:1
  2Adam/conv3d_17/kernel/v
!: 2Adam/conv3d_17/bias/v
/:- 2#Adam/batch_normalization_42/gamma/v
.:, 2"Adam/batch_normalization_42/beta/v
3:1
  2Adam/conv3d_18/kernel/v
!: 2Adam/conv3d_18/bias/v
/:- 2#Adam/batch_normalization_43/gamma/v
.:, 2"Adam/batch_normalization_43/beta/v
3:1
  2Adam/conv3d_19/kernel/v
!: 2Adam/conv3d_19/bias/v
/:- 2#Adam/batch_normalization_44/gamma/v
.:, 2"Adam/batch_normalization_44/beta/v
(:&
?
?2Adam/dense_16/kernel/v
!:?2Adam/dense_16/bias/v
':%	?2Adam/dense_17/kernel/v
 :2Adam/dense_17/bias/v
6:4
 2Adam/conv3d_15/kernel/vhat
$:" 2Adam/conv3d_15/bias/vhat
2:0 2&Adam/batch_normalization_40/gamma/vhat
1:/ 2%Adam/batch_normalization_40/beta/vhat
6:4
  2Adam/conv3d_16/kernel/vhat
$:" 2Adam/conv3d_16/bias/vhat
2:0 2&Adam/batch_normalization_41/gamma/vhat
1:/ 2%Adam/batch_normalization_41/beta/vhat
6:4
  2Adam/conv3d_17/kernel/vhat
$:" 2Adam/conv3d_17/bias/vhat
2:0 2&Adam/batch_normalization_42/gamma/vhat
1:/ 2%Adam/batch_normalization_42/beta/vhat
6:4
  2Adam/conv3d_18/kernel/vhat
$:" 2Adam/conv3d_18/bias/vhat
2:0 2&Adam/batch_normalization_43/gamma/vhat
1:/ 2%Adam/batch_normalization_43/beta/vhat
6:4
  2Adam/conv3d_19/kernel/vhat
$:" 2Adam/conv3d_19/bias/vhat
2:0 2&Adam/batch_normalization_44/gamma/vhat
1:/ 2%Adam/batch_normalization_44/beta/vhat
+:)
?
?2Adam/dense_16/kernel/vhat
$:"?2Adam/dense_16/bias/vhat
*:(	?2Adam/dense_17/kernel/vhat
#:!2Adam/dense_17/bias/vhat
?2?
!__inference__wrapped_model_418133?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *:?7
5?2
conv3d_15_input?????????
@@
?2?
-__inference_sequential_9_layer_call_fn_419421
-__inference_sequential_9_layer_call_fn_420588
-__inference_sequential_9_layer_call_fn_420661
-__inference_sequential_9_layer_call_fn_420178?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_9_layer_call_and_return_conditional_losses_420827
H__inference_sequential_9_layer_call_and_return_conditional_losses_421077
H__inference_sequential_9_layer_call_and_return_conditional_losses_420294
H__inference_sequential_9_layer_call_and_return_conditional_losses_420410?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv3d_15_layer_call_fn_421086?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv3d_15_layer_call_and_return_conditional_losses_421097?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_40_layer_call_fn_421110
7__inference_batch_normalization_40_layer_call_fn_421123
7__inference_batch_normalization_40_layer_call_fn_421136
7__inference_batch_normalization_40_layer_call_fn_421149?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421169
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421203
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421223
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421257?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_max_pooling3d_30_layer_call_fn_418307?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
L__inference_max_pooling3d_30_layer_call_and_return_conditional_losses_418301?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
*__inference_conv3d_16_layer_call_fn_421272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv3d_16_layer_call_and_return_conditional_losses_421289?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_41_layer_call_fn_421302
7__inference_batch_normalization_41_layer_call_fn_421315
7__inference_batch_normalization_41_layer_call_fn_421328
7__inference_batch_normalization_41_layer_call_fn_421341?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421361
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421395
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421415
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421449?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_max_pooling3d_31_layer_call_fn_418481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
L__inference_max_pooling3d_31_layer_call_and_return_conditional_losses_418475?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
*__inference_conv3d_17_layer_call_fn_421464?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv3d_17_layer_call_and_return_conditional_losses_421481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_42_layer_call_fn_421494
7__inference_batch_normalization_42_layer_call_fn_421507
7__inference_batch_normalization_42_layer_call_fn_421520
7__inference_batch_normalization_42_layer_call_fn_421533?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421553
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421587
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421607
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421641?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_max_pooling3d_32_layer_call_fn_418655?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
L__inference_max_pooling3d_32_layer_call_and_return_conditional_losses_418649?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
*__inference_conv3d_18_layer_call_fn_421656?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv3d_18_layer_call_and_return_conditional_losses_421673?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_43_layer_call_fn_421686
7__inference_batch_normalization_43_layer_call_fn_421699
7__inference_batch_normalization_43_layer_call_fn_421712
7__inference_batch_normalization_43_layer_call_fn_421725?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421745
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421779
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421799
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421833?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_max_pooling3d_33_layer_call_fn_418829?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
L__inference_max_pooling3d_33_layer_call_and_return_conditional_losses_418823?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
*__inference_conv3d_19_layer_call_fn_421848?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv3d_19_layer_call_and_return_conditional_losses_421865?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_44_layer_call_fn_421878
7__inference_batch_normalization_44_layer_call_fn_421891
7__inference_batch_normalization_44_layer_call_fn_421904
7__inference_batch_normalization_44_layer_call_fn_421917?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_421937
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_421971
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_421991
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_422025?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_max_pooling3d_34_layer_call_fn_419003?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
L__inference_max_pooling3d_34_layer_call_and_return_conditional_losses_418997?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
*__inference_flatten_8_layer_call_fn_422030?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_8_layer_call_and_return_conditional_losses_422036?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dropout_16_layer_call_fn_422041
+__inference_dropout_16_layer_call_fn_422046?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_16_layer_call_and_return_conditional_losses_422051
F__inference_dropout_16_layer_call_and_return_conditional_losses_422063?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_16_layer_call_fn_422072?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_16_layer_call_and_return_conditional_losses_422083?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dropout_17_layer_call_fn_422088
+__inference_dropout_17_layer_call_fn_422093?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_17_layer_call_and_return_conditional_losses_422098
F__inference_dropout_17_layer_call_and_return_conditional_losses_422110?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_17_layer_call_fn_422119?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_17_layer_call_and_return_conditional_losses_422130?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_422141?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_422152?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_422163?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_422174?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
$__inference_signature_wrapper_420515conv3d_15_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_418133?&%"$#./8576ABKHJITU^[]\ghqnpo????D?A
:?7
5?2
conv3d_15_input?????????
@@
? "3?0
.
dense_17"?
dense_17??????????
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421169?%"$#Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421203?$%"#Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421223z%"$#??<
5?2
,?)
inputs?????????
@@ 
p 
? "1?.
'?$
0?????????
@@ 
? ?
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_421257z$%"#??<
5?2
,?)
inputs?????????
@@ 
p
? "1?.
'?$
0?????????
@@ 
? ?
7__inference_batch_normalization_40_layer_call_fn_421110?%"$#Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_40_layer_call_fn_421123?$%"#Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_40_layer_call_fn_421136m%"$#??<
5?2
,?)
inputs?????????
@@ 
p 
? "$?!?????????
@@ ?
7__inference_batch_normalization_40_layer_call_fn_421149m$%"#??<
5?2
,?)
inputs?????????
@@ 
p
? "$?!?????????
@@ ?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421361?8576Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421395?7856Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421415z8576??<
5?2
,?)
inputs?????????
   
p 
? "1?.
'?$
0?????????
   
? ?
R__inference_batch_normalization_41_layer_call_and_return_conditional_losses_421449z7856??<
5?2
,?)
inputs?????????
   
p
? "1?.
'?$
0?????????
   
? ?
7__inference_batch_normalization_41_layer_call_fn_421302?8576Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_41_layer_call_fn_421315?7856Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_41_layer_call_fn_421328m8576??<
5?2
,?)
inputs?????????
   
p 
? "$?!?????????
   ?
7__inference_batch_normalization_41_layer_call_fn_421341m7856??<
5?2
,?)
inputs?????????
   
p
? "$?!?????????
   ?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421553?KHJIZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421587?JKHIZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421607zKHJI??<
5?2
,?)
inputs?????????
 
p 
? "1?.
'?$
0?????????
 
? ?
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_421641zJKHI??<
5?2
,?)
inputs?????????
 
p
? "1?.
'?$
0?????????
 
? ?
7__inference_batch_normalization_42_layer_call_fn_421494?KHJIZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_42_layer_call_fn_421507?JKHIZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_42_layer_call_fn_421520mKHJI??<
5?2
,?)
inputs?????????
 
p 
? "$?!?????????
 ?
7__inference_batch_normalization_42_layer_call_fn_421533mJKHI??<
5?2
,?)
inputs?????????
 
p
? "$?!?????????
 ?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421745?^[]\Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421779?]^[\Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421799z^[]\??<
5?2
,?)
inputs?????????
 
p 
? "1?.
'?$
0?????????
 
? ?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_421833z]^[\??<
5?2
,?)
inputs?????????
 
p
? "1?.
'?$
0?????????
 
? ?
7__inference_batch_normalization_43_layer_call_fn_421686?^[]\Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_43_layer_call_fn_421699?]^[\Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_43_layer_call_fn_421712m^[]\??<
5?2
,?)
inputs?????????
 
p 
? "$?!?????????
 ?
7__inference_batch_normalization_43_layer_call_fn_421725m]^[\??<
5?2
,?)
inputs?????????
 
p
? "$?!?????????
 ?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_421937?qnpoZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_421971?pqnoZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_421991zqnpo??<
5?2
,?)
inputs?????????
 
p 
? "1?.
'?$
0?????????
 
? ?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_422025zpqno??<
5?2
,?)
inputs?????????
 
p
? "1?.
'?$
0?????????
 
? ?
7__inference_batch_normalization_44_layer_call_fn_421878?qnpoZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_44_layer_call_fn_421891?pqnoZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_44_layer_call_fn_421904mqnpo??<
5?2
,?)
inputs?????????
 
p 
? "$?!?????????
 ?
7__inference_batch_normalization_44_layer_call_fn_421917mpqno??<
5?2
,?)
inputs?????????
 
p
? "$?!?????????
 ?
E__inference_conv3d_15_layer_call_and_return_conditional_losses_421097t;?8
1?.
,?)
inputs?????????
@@
? "1?.
'?$
0?????????
@@ 
? ?
*__inference_conv3d_15_layer_call_fn_421086g;?8
1?.
,?)
inputs?????????
@@
? "$?!?????????
@@ ?
E__inference_conv3d_16_layer_call_and_return_conditional_losses_421289t./;?8
1?.
,?)
inputs?????????
   
? "1?.
'?$
0?????????
   
? ?
*__inference_conv3d_16_layer_call_fn_421272g./;?8
1?.
,?)
inputs?????????
   
? "$?!?????????
   ?
E__inference_conv3d_17_layer_call_and_return_conditional_losses_421481tAB;?8
1?.
,?)
inputs?????????
 
? "1?.
'?$
0?????????
 
? ?
*__inference_conv3d_17_layer_call_fn_421464gAB;?8
1?.
,?)
inputs?????????
 
? "$?!?????????
 ?
E__inference_conv3d_18_layer_call_and_return_conditional_losses_421673tTU;?8
1?.
,?)
inputs?????????
 
? "1?.
'?$
0?????????
 
? ?
*__inference_conv3d_18_layer_call_fn_421656gTU;?8
1?.
,?)
inputs?????????
 
? "$?!?????????
 ?
E__inference_conv3d_19_layer_call_and_return_conditional_losses_421865tgh;?8
1?.
,?)
inputs?????????
 
? "1?.
'?$
0?????????
 
? ?
*__inference_conv3d_19_layer_call_fn_421848ggh;?8
1?.
,?)
inputs?????????
 
? "$?!?????????
 ?
D__inference_dense_16_layer_call_and_return_conditional_losses_422083`??0?-
&?#
!?
inputs??????????

? "&?#
?
0??????????
? ?
)__inference_dense_16_layer_call_fn_422072S??0?-
&?#
!?
inputs??????????

? "????????????
D__inference_dense_17_layer_call_and_return_conditional_losses_422130_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? 
)__inference_dense_17_layer_call_fn_422119R??0?-
&?#
!?
inputs??????????
? "???????????
F__inference_dropout_16_layer_call_and_return_conditional_losses_422051^4?1
*?'
!?
inputs??????????

p 
? "&?#
?
0??????????

? ?
F__inference_dropout_16_layer_call_and_return_conditional_losses_422063^4?1
*?'
!?
inputs??????????

p
? "&?#
?
0??????????

? ?
+__inference_dropout_16_layer_call_fn_422041Q4?1
*?'
!?
inputs??????????

p 
? "???????????
?
+__inference_dropout_16_layer_call_fn_422046Q4?1
*?'
!?
inputs??????????

p
? "???????????
?
F__inference_dropout_17_layer_call_and_return_conditional_losses_422098^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
F__inference_dropout_17_layer_call_and_return_conditional_losses_422110^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
+__inference_dropout_17_layer_call_fn_422088Q4?1
*?'
!?
inputs??????????
p 
? "????????????
+__inference_dropout_17_layer_call_fn_422093Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_flatten_8_layer_call_and_return_conditional_losses_422036e;?8
1?.
,?)
inputs?????????
 
? "&?#
?
0??????????

? ?
*__inference_flatten_8_layer_call_fn_422030X;?8
1?.
,?)
inputs?????????
 
? "???????????
;
__inference_loss_fn_0_422141.?

? 
? "? ;
__inference_loss_fn_1_422152A?

? 
? "? ;
__inference_loss_fn_2_422163T?

? 
? "? ;
__inference_loss_fn_3_422174g?

? 
? "? ?
L__inference_max_pooling3d_30_layer_call_and_return_conditional_losses_418301?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
1__inference_max_pooling3d_30_layer_call_fn_418307?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
L__inference_max_pooling3d_31_layer_call_and_return_conditional_losses_418475?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
1__inference_max_pooling3d_31_layer_call_fn_418481?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
L__inference_max_pooling3d_32_layer_call_and_return_conditional_losses_418649?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
1__inference_max_pooling3d_32_layer_call_fn_418655?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
L__inference_max_pooling3d_33_layer_call_and_return_conditional_losses_418823?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
1__inference_max_pooling3d_33_layer_call_fn_418829?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
L__inference_max_pooling3d_34_layer_call_and_return_conditional_losses_418997?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
1__inference_max_pooling3d_34_layer_call_fn_419003?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
H__inference_sequential_9_layer_call_and_return_conditional_losses_420294?&%"$#./8576ABKHJITU^[]\ghqnpo????L?I
B??
5?2
conv3d_15_input?????????
@@
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_420410?&$%"#./7856ABJKHITU]^[\ghpqno????L?I
B??
5?2
conv3d_15_input?????????
@@
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_420827?&%"$#./8576ABKHJITU^[]\ghqnpo????C?@
9?6
,?)
inputs?????????
@@
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_421077?&$%"#./7856ABJKHITU]^[\ghpqno????C?@
9?6
,?)
inputs?????????
@@
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_9_layer_call_fn_419421?&%"$#./8576ABKHJITU^[]\ghqnpo????L?I
B??
5?2
conv3d_15_input?????????
@@
p 

 
? "???????????
-__inference_sequential_9_layer_call_fn_420178?&$%"#./7856ABJKHITU]^[\ghpqno????L?I
B??
5?2
conv3d_15_input?????????
@@
p

 
? "???????????
-__inference_sequential_9_layer_call_fn_420588?&%"$#./8576ABKHJITU^[]\ghqnpo????C?@
9?6
,?)
inputs?????????
@@
p 

 
? "???????????
-__inference_sequential_9_layer_call_fn_420661?&$%"#./7856ABJKHITU]^[\ghpqno????C?@
9?6
,?)
inputs?????????
@@
p

 
? "???????????
$__inference_signature_wrapper_420515?&%"$#./8576ABKHJITU^[]\ghqnpo????W?T
? 
M?J
H
conv3d_15_input5?2
conv3d_15_input?????????
@@"3?0
.
dense_17"?
dense_17?????????
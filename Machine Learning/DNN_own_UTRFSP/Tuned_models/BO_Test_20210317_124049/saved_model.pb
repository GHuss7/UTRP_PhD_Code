╡╬
оГ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
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
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
╛
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
executor_typestring И
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.0-49-g85c8b2a817f8ьТ
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ДP*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	ДP*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:P*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:PP*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:P*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:P*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
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
y
dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ДP*
shared_namedense/kernel/m
r
"dense/kernel/m/Read/ReadVariableOpReadVariableOpdense/kernel/m*
_output_shapes
:	ДP*
dtype0
p
dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense/bias/m
i
 dense/bias/m/Read/ReadVariableOpReadVariableOpdense/bias/m*
_output_shapes
:P*
dtype0
|
dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*!
shared_namedense_1/kernel/m
u
$dense_1/kernel/m/Read/ReadVariableOpReadVariableOpdense_1/kernel/m*
_output_shapes

:PP*
dtype0
t
dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_1/bias/m
m
"dense_1/bias/m/Read/ReadVariableOpReadVariableOpdense_1/bias/m*
_output_shapes
:P*
dtype0
|
dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*!
shared_namedense_2/kernel/m
u
$dense_2/kernel/m/Read/ReadVariableOpReadVariableOpdense_2/kernel/m*
_output_shapes

:P*
dtype0
t
dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias/m
m
"dense_2/bias/m/Read/ReadVariableOpReadVariableOpdense_2/bias/m*
_output_shapes
:*
dtype0
y
dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ДP*
shared_namedense/kernel/v
r
"dense/kernel/v/Read/ReadVariableOpReadVariableOpdense/kernel/v*
_output_shapes
:	ДP*
dtype0
p
dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense/bias/v
i
 dense/bias/v/Read/ReadVariableOpReadVariableOpdense/bias/v*
_output_shapes
:P*
dtype0
|
dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*!
shared_namedense_1/kernel/v
u
$dense_1/kernel/v/Read/ReadVariableOpReadVariableOpdense_1/kernel/v*
_output_shapes

:PP*
dtype0
t
dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_1/bias/v
m
"dense_1/bias/v/Read/ReadVariableOpReadVariableOpdense_1/bias/v*
_output_shapes
:P*
dtype0
|
dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*!
shared_namedense_2/kernel/v
u
$dense_2/kernel/v/Read/ReadVariableOpReadVariableOpdense_2/kernel/v*
_output_shapes

:P*
dtype0
t
dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias/v
m
"dense_2/bias/v/Read/ReadVariableOpReadVariableOpdense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
├%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*■$
valueЇ$Bё$ Bъ$
А
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
lmOmPmQmR mS!mTvUvVvWvX vY!vZ
*
0
1
2
3
 4
!5
 
*
0
1
2
3
 4
!5
н
&non_trainable_variables
	variables
'layer_regularization_losses
(layer_metrics
)metrics
regularization_losses
	trainable_variables

*layers
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н
+non_trainable_variables
	variables
,layer_metrics
trainable_variables
-metrics
regularization_losses

.layers
/layer_regularization_losses
 
 
 
н
0non_trainable_variables
	variables
1layer_metrics
trainable_variables
2metrics
regularization_losses

3layers
4layer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н
5non_trainable_variables
	variables
6layer_metrics
trainable_variables
7metrics
regularization_losses

8layers
9layer_regularization_losses
 
 
 
н
:non_trainable_variables
	variables
;layer_metrics
trainable_variables
<metrics
regularization_losses

=layers
>layer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
н
?non_trainable_variables
"	variables
@layer_metrics
#trainable_variables
Ametrics
$regularization_losses

Blayers
Clayer_regularization_losses
 
 
 

D0
E1
#
0
1
2
3
4
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
4
	Ftotal
	Gcount
H	variables
I	keras_api
D
	Jtotal
	Kcount
L
_fn_kwargs
M	variables
N	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

H	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

M	variables
vt
VARIABLE_VALUEdense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEdense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А
serving_default_dense_inputPlaceholder*(
_output_shapes
:         Д*
dtype0*
shape:         Д
Щ
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_inputdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_8528299
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
д
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"dense/kernel/m/Read/ReadVariableOp dense/bias/m/Read/ReadVariableOp$dense_1/kernel/m/Read/ReadVariableOp"dense_1/bias/m/Read/ReadVariableOp$dense_2/kernel/m/Read/ReadVariableOp"dense_2/bias/m/Read/ReadVariableOp"dense/kernel/v/Read/ReadVariableOp dense/bias/v/Read/ReadVariableOp$dense_1/kernel/v/Read/ReadVariableOp"dense_1/bias/v/Read/ReadVariableOp$dense_2/kernel/v/Read/ReadVariableOp"dense_2/bias/v/Read/ReadVariableOpConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_save_8528604
ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biastotalcounttotal_1count_1dense/kernel/mdense/bias/mdense_1/kernel/mdense_1/bias/mdense_2/kernel/mdense_2/bias/mdense/kernel/vdense/bias/vdense_1/kernel/vdense_1/bias/vdense_2/kernel/vdense_2/bias/v*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__traced_restore_8528680╚┤
Ї
┬
,__inference_sequential_layer_call_fn_8528242
dense_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_85282272
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:         Д
%
_user_specified_namedense_input
║^
р

#__inference__traced_restore_8528680
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias
assignvariableop_6_total
assignvariableop_7_count
assignvariableop_8_total_1
assignvariableop_9_count_1&
"assignvariableop_10_dense_kernel_m$
 assignvariableop_11_dense_bias_m(
$assignvariableop_12_dense_1_kernel_m&
"assignvariableop_13_dense_1_bias_m(
$assignvariableop_14_dense_2_kernel_m&
"assignvariableop_15_dense_2_bias_m&
"assignvariableop_16_dense_kernel_v$
 assignvariableop_17_dense_bias_v(
$assignvariableop_18_dense_1_kernel_v&
"assignvariableop_19_dense_1_bias_v(
$assignvariableop_20_dense_2_kernel_v&
"assignvariableop_21_dense_2_bias_v
identity_23ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9н
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╣
valueпBмB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╝
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЮ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЬ
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1в
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ж
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3д
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ж
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5д
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Э
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Э
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Я
AssignVariableOp_8AssignVariableOpassignvariableop_8_total_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Я
AssignVariableOp_9AssignVariableOpassignvariableop_9_count_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10к
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_kernel_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11и
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_bias_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12м
AssignVariableOp_12AssignVariableOp$assignvariableop_12_dense_1_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13к
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_1_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14м
AssignVariableOp_14AssignVariableOp$assignvariableop_14_dense_2_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15к
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_2_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16к
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_kernel_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17и
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_bias_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18м
AssignVariableOp_18AssignVariableOp$assignvariableop_18_dense_1_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19к
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_1_bias_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20м
AssignVariableOp_20AssignVariableOp$assignvariableop_20_dense_2_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21к
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_2_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp┬
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22╡
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
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
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
│5
╤
 __inference__traced_save_8528604
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_dense_kernel_m_read_readvariableop+
'savev2_dense_bias_m_read_readvariableop/
+savev2_dense_1_kernel_m_read_readvariableop-
)savev2_dense_1_bias_m_read_readvariableop/
+savev2_dense_2_kernel_m_read_readvariableop-
)savev2_dense_2_bias_m_read_readvariableop-
)savev2_dense_kernel_v_read_readvariableop+
'savev2_dense_bias_v_read_readvariableop/
+savev2_dense_1_kernel_v_read_readvariableop-
)savev2_dense_1_bias_v_read_readvariableop/
+savev2_dense_2_kernel_v_read_readvariableop-
)savev2_dense_2_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameз
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╣
valueпBмB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╢
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices▄
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_dense_kernel_m_read_readvariableop'savev2_dense_bias_m_read_readvariableop+savev2_dense_1_kernel_m_read_readvariableop)savev2_dense_1_bias_m_read_readvariableop+savev2_dense_2_kernel_m_read_readvariableop)savev2_dense_2_bias_m_read_readvariableop)savev2_dense_kernel_v_read_readvariableop'savev2_dense_bias_v_read_readvariableop+savev2_dense_1_kernel_v_read_readvariableop)savev2_dense_1_bias_v_read_readvariableop+savev2_dense_2_kernel_v_read_readvariableop)savev2_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*┤
_input_shapesв
Я: :	ДP:P:PP:P:P:: : : : :	ДP:P:PP:P:P::	ДP:P:PP:P:P:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ДP: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:P: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :%!

_output_shapes
:	ДP: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:P: 

_output_shapes
::%!

_output_shapes
:	ДP: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:P: 

_output_shapes
::

_output_shapes
: 
√
К
G__inference_sequential_layer_call_and_return_conditional_losses_8528227

inputs
dense_8528209
dense_8528211
dense_1_8528215
dense_1_8528217
dense_2_8528221
dense_2_8528223
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallИ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8528209dense_8528211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_85280512
dense/StatefulPartitionedCallК
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_85280792!
dropout/StatefulPartitionedCall┤
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_8528215dense_1_8528217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_85281082!
dense_1/StatefulPartitionedCall┤
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_85281362#
!dropout_1/StatefulPartitionedCall╢
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_8528221dense_2_8528223*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_85281652!
dense_2/StatefulPartitionedCallж
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs
в
d
+__inference_dropout_1_layer_call_fn_8528490

inputs
identityИвStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_85281362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*&
_input_shapes
:         P22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
В
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_8528136

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*&
_input_shapes
:         P:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
я	
█
B__inference_dense_layer_call_and_return_conditional_losses_8528412

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ДP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         P2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Д::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs
╟
b
D__inference_dropout_layer_call_and_return_conditional_losses_8528438

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         P2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         P:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
К
П
G__inference_sequential_layer_call_and_return_conditional_losses_8528182
dense_input
dense_8528062
dense_8528064
dense_1_8528119
dense_1_8528121
dense_2_8528176
dense_2_8528178
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallН
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_8528062dense_8528064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_85280512
dense/StatefulPartitionedCallК
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_85280792!
dropout/StatefulPartitionedCall┤
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_8528119dense_1_8528121*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_85281082!
dense_1/StatefulPartitionedCall┤
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_85281362#
!dropout_1/StatefulPartitionedCall╢
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_8528176dense_2_8528178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_85281652!
dense_2/StatefulPartitionedCallж
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:U Q
(
_output_shapes
:         Д
%
_user_specified_namedense_input
Ы1
м
G__inference_sequential_layer_call_and_return_conditional_losses_8528340

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ДP*
dtype02
dense/MatMul/ReadVariableOpЕ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         P2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/dropout/ConstЭ
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:         P2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape╠
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:         P*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2 
dropout/dropout/GreaterEqual/y▐
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         P2
dropout/dropout/GreaterEqualЧ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         P2
dropout/dropout/CastЪ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:         P2
dropout/dropout/Mul_1е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
dense_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_1/dropout/Constе
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:         P2
dropout_1/dropout/Mul|
dropout_1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape╥
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:         P*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2"
 dropout_1/dropout/GreaterEqual/yц
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         P2 
dropout_1/dropout/GreaterEqualЭ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         P2
dropout_1/dropout/Castв
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:         P2
dropout_1/dropout/Mul_1е
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
dense_2/MatMul/ReadVariableOpа
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_2/Reluн
IdentityIdentitydense_2/Relu:activations:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs
ю	
▌
D__inference_dense_2_layer_call_and_return_conditional_losses_8528165

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
Ю
b
)__inference_dropout_layer_call_fn_8528443

inputs
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_85280792
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*&
_input_shapes
:         P22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
Б
─
G__inference_sequential_layer_call_and_return_conditional_losses_8528265

inputs
dense_8528247
dense_8528249
dense_1_8528253
dense_1_8528255
dense_2_8528259
dense_2_8528261
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallИ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8528247dense_8528249*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_85280512
dense/StatefulPartitionedCallЄ
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_85280842
dropout/PartitionedCallм
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_8528253dense_1_8528255*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_85281082!
dense_1/StatefulPartitionedCall·
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_85281412
dropout_1/PartitionedCallо
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_8528259dense_2_8528261*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_85281652!
dense_2/StatefulPartitionedCallр
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs
х
╜
,__inference_sequential_layer_call_fn_8528401

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_85282652
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs
А
c
D__inference_dropout_layer_call_and_return_conditional_losses_8528433

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*&
_input_shapes
:         P:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
Ї
м
G__inference_sequential_layer_call_and_return_conditional_losses_8528367

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ДP*
dtype02
dense/MatMul/ReadVariableOpЕ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         P2

dense/Relu|
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:         P2
dropout/Identityе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
dense_1/ReluВ
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:         P2
dropout_1/Identityе
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
dense_2/MatMul/ReadVariableOpа
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_2/Reluн
IdentityIdentitydense_2/Relu:activations:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs
Ї
┬
,__inference_sequential_layer_call_fn_8528280
dense_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_85282652
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:         Д
%
_user_specified_namedense_input
я	
█
B__inference_dense_layer_call_and_return_conditional_losses_8528051

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ДP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         P2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Д::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs
В
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_8528480

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*&
_input_shapes
:         P:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
х
╜
,__inference_sequential_layer_call_fn_8528384

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_85282272
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs
Р
╔
G__inference_sequential_layer_call_and_return_conditional_losses_8528203
dense_input
dense_8528185
dense_8528187
dense_1_8528191
dense_1_8528193
dense_2_8528197
dense_2_8528199
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallН
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_8528185dense_8528187*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_85280512
dense/StatefulPartitionedCallЄ
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_85280842
dropout/PartitionedCallм
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_8528191dense_1_8528193*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_85281082!
dense_1/StatefulPartitionedCall·
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_85281412
dropout_1/PartitionedCallо
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_8528197dense_2_8528199*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_85281652!
dense_2/StatefulPartitionedCallр
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:U Q
(
_output_shapes
:         Д
%
_user_specified_namedense_input
А
c
D__inference_dropout_layer_call_and_return_conditional_losses_8528079

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*&
_input_shapes
:         P:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
Т
E
)__inference_dropout_layer_call_fn_8528448

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_85280842
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*&
_input_shapes
:         P:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
╔
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_8528485

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         P2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         P:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
╚
╗
%__inference_signature_wrapper_8528299
dense_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_85280362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:         Д
%
_user_specified_namedense_input
ю	
▌
D__inference_dense_1_layer_call_and_return_conditional_losses_8528459

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         P2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
┌
|
'__inference_dense_layer_call_fn_8528421

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_85280512
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Д::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs
Ц
G
+__inference_dropout_1_layer_call_fn_8528495

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_85281412
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*&
_input_shapes
:         P:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
▄
~
)__inference_dense_2_layer_call_fn_8528515

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_85281652
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
Я&
Р
"__inference__wrapped_model_8528036
dense_input3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identityИв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв)sequential/dense_1/BiasAdd/ReadVariableOpв(sequential/dense_1/MatMul/ReadVariableOpв)sequential/dense_2/BiasAdd/ReadVariableOpв(sequential/dense_2/MatMul/ReadVariableOp┴
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	ДP*
dtype02(
&sequential/dense/MatMul/ReadVariableOpл
sequential/dense/MatMulMatMuldense_input.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
sequential/dense/MatMul┐
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp┼
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
sequential/dense/BiasAddЛ
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
sequential/dense/ReluЭ
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*'
_output_shapes
:         P2
sequential/dropout/Identity╞
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp╩
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
sequential/dense_1/MatMul┼
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp═
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
sequential/dense_1/BiasAddС
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
sequential/dense_1/Reluг
sequential/dropout_1/IdentityIdentity%sequential/dense_1/Relu:activations:0*
T0*'
_output_shapes
:         P2
sequential/dropout_1/Identity╞
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp╠
sequential/dense_2/MatMulMatMul&sequential/dropout_1/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/dense_2/MatMul┼
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp═
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/dense_2/BiasAddС
sequential/dense_2/ReluRelu#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential/dense_2/Relu·
IdentityIdentity%sequential/dense_2/Relu:activations:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         Д::::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp:U Q
(
_output_shapes
:         Д
%
_user_specified_namedense_input
╔
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_8528141

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         P2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         P:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
▄
~
)__inference_dense_1_layer_call_fn_8528468

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_85281082
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
ю	
▌
D__inference_dense_1_layer_call_and_return_conditional_losses_8528108

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         P2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
ю	
▌
D__inference_dense_2_layer_call_and_return_conditional_losses_8528506

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs
╟
b
D__inference_dropout_layer_call_and_return_conditional_losses_8528084

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         P2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         P:O K
'
_output_shapes
:         P
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
D
dense_input5
serving_default_dense_input:0         Д;
dense_20
StatefulPartitionedCall:0         tensorflow/serving/predict:лл
о'
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
[__call__
*\&call_and_return_all_conditional_losses
]_default_save_signature"╘$
_tf_keras_sequential╡${"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 132]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 132]}, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 132}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 132]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 132]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 132]}, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
у

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
^__call__
*_&call_and_return_all_conditional_losses"╛
_tf_keras_layerд{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 132]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 132]}, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 132}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 132]}}
с
	variables
trainable_variables
regularization_losses
	keras_api
`__call__
*a&call_and_return_all_conditional_losses"╥
_tf_keras_layer╕{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
Ё

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
b__call__
*c&call_and_return_all_conditional_losses"╦
_tf_keras_layer▒{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
х
	variables
trainable_variables
regularization_losses
	keras_api
d__call__
*e&call_and_return_all_conditional_losses"╓
_tf_keras_layer╝{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
я

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
f__call__
*g&call_and_return_all_conditional_losses"╩
_tf_keras_layer░{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
mOmPmQmR mS!mTvUvVvWvX vY!vZ"
	optimizer
J
0
1
2
3
 4
!5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
╩
&non_trainable_variables
	variables
'layer_regularization_losses
(layer_metrics
)metrics
regularization_losses
	trainable_variables

*layers
[__call__
]_default_save_signature
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
,
hserving_default"
signature_map
:	ДP2dense/kernel
:P2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
+non_trainable_variables
	variables
,layer_metrics
trainable_variables
-metrics
regularization_losses

.layers
/layer_regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
0non_trainable_variables
	variables
1layer_metrics
trainable_variables
2metrics
regularization_losses

3layers
4layer_regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 :PP2dense_1/kernel
:P2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
5non_trainable_variables
	variables
6layer_metrics
trainable_variables
7metrics
regularization_losses

8layers
9layer_regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
:non_trainable_variables
	variables
;layer_metrics
trainable_variables
<metrics
regularization_losses

=layers
>layer_regularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 :P2dense_2/kernel
:2dense_2/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
?non_trainable_variables
"	variables
@layer_metrics
#trainable_variables
Ametrics
$regularization_losses

Blayers
Clayer_regularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
D0
E1"
trackable_list_wrapper
C
0
1
2
3
4"
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
╗
	Ftotal
	Gcount
H	variables
I	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Ї
	Jtotal
	Kcount
L
_fn_kwargs
M	variables
N	keras_api"н
_tf_keras_metricТ{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
:  (2total
:  (2count
.
F0
G1"
trackable_list_wrapper
-
H	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
:	ДP2dense/kernel/m
:P2dense/bias/m
 :PP2dense_1/kernel/m
:P2dense_1/bias/m
 :P2dense_2/kernel/m
:2dense_2/bias/m
:	ДP2dense/kernel/v
:P2dense/bias/v
 :PP2dense_1/kernel/v
:P2dense_1/bias/v
 :P2dense_2/kernel/v
:2dense_2/bias/v
■2√
,__inference_sequential_layer_call_fn_8528384
,__inference_sequential_layer_call_fn_8528242
,__inference_sequential_layer_call_fn_8528280
,__inference_sequential_layer_call_fn_8528401└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
G__inference_sequential_layer_call_and_return_conditional_losses_8528367
G__inference_sequential_layer_call_and_return_conditional_losses_8528340
G__inference_sequential_layer_call_and_return_conditional_losses_8528182
G__inference_sequential_layer_call_and_return_conditional_losses_8528203└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
х2т
"__inference__wrapped_model_8528036╗
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *+в(
&К#
dense_input         Д
╤2╬
'__inference_dense_layer_call_fn_8528421в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_layer_call_and_return_conditional_losses_8528412в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Р2Н
)__inference_dropout_layer_call_fn_8528443
)__inference_dropout_layer_call_fn_8528448┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╞2├
D__inference_dropout_layer_call_and_return_conditional_losses_8528433
D__inference_dropout_layer_call_and_return_conditional_losses_8528438┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╙2╨
)__inference_dense_1_layer_call_fn_8528468в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_dense_1_layer_call_and_return_conditional_losses_8528459в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ф2С
+__inference_dropout_1_layer_call_fn_8528495
+__inference_dropout_1_layer_call_fn_8528490┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_1_layer_call_and_return_conditional_losses_8528480
F__inference_dropout_1_layer_call_and_return_conditional_losses_8528485┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╙2╨
)__inference_dense_2_layer_call_fn_8528515в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_dense_2_layer_call_and_return_conditional_losses_8528506в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨B═
%__inference_signature_wrapper_8528299dense_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 Ш
"__inference__wrapped_model_8528036r !5в2
+в(
&К#
dense_input         Д
к "1к.
,
dense_2!К
dense_2         д
D__inference_dense_1_layer_call_and_return_conditional_losses_8528459\/в,
%в"
 К
inputs         P
к "%в"
К
0         P
Ъ |
)__inference_dense_1_layer_call_fn_8528468O/в,
%в"
 К
inputs         P
к "К         Pд
D__inference_dense_2_layer_call_and_return_conditional_losses_8528506\ !/в,
%в"
 К
inputs         P
к "%в"
К
0         
Ъ |
)__inference_dense_2_layer_call_fn_8528515O !/в,
%в"
 К
inputs         P
к "К         г
B__inference_dense_layer_call_and_return_conditional_losses_8528412]0в-
&в#
!К
inputs         Д
к "%в"
К
0         P
Ъ {
'__inference_dense_layer_call_fn_8528421P0в-
&в#
!К
inputs         Д
к "К         Pж
F__inference_dropout_1_layer_call_and_return_conditional_losses_8528480\3в0
)в&
 К
inputs         P
p
к "%в"
К
0         P
Ъ ж
F__inference_dropout_1_layer_call_and_return_conditional_losses_8528485\3в0
)в&
 К
inputs         P
p 
к "%в"
К
0         P
Ъ ~
+__inference_dropout_1_layer_call_fn_8528490O3в0
)в&
 К
inputs         P
p
к "К         P~
+__inference_dropout_1_layer_call_fn_8528495O3в0
)в&
 К
inputs         P
p 
к "К         Pд
D__inference_dropout_layer_call_and_return_conditional_losses_8528433\3в0
)в&
 К
inputs         P
p
к "%в"
К
0         P
Ъ д
D__inference_dropout_layer_call_and_return_conditional_losses_8528438\3в0
)в&
 К
inputs         P
p 
к "%в"
К
0         P
Ъ |
)__inference_dropout_layer_call_fn_8528443O3в0
)в&
 К
inputs         P
p
к "К         P|
)__inference_dropout_layer_call_fn_8528448O3в0
)в&
 К
inputs         P
p 
к "К         P╣
G__inference_sequential_layer_call_and_return_conditional_losses_8528182n !=в:
3в0
&К#
dense_input         Д
p

 
к "%в"
К
0         
Ъ ╣
G__inference_sequential_layer_call_and_return_conditional_losses_8528203n !=в:
3в0
&К#
dense_input         Д
p 

 
к "%в"
К
0         
Ъ ┤
G__inference_sequential_layer_call_and_return_conditional_losses_8528340i !8в5
.в+
!К
inputs         Д
p

 
к "%в"
К
0         
Ъ ┤
G__inference_sequential_layer_call_and_return_conditional_losses_8528367i !8в5
.в+
!К
inputs         Д
p 

 
к "%в"
К
0         
Ъ С
,__inference_sequential_layer_call_fn_8528242a !=в:
3в0
&К#
dense_input         Д
p

 
к "К         С
,__inference_sequential_layer_call_fn_8528280a !=в:
3в0
&К#
dense_input         Д
p 

 
к "К         М
,__inference_sequential_layer_call_fn_8528384\ !8в5
.в+
!К
inputs         Д
p

 
к "К         М
,__inference_sequential_layer_call_fn_8528401\ !8в5
.в+
!К
inputs         Д
p 

 
к "К         л
%__inference_signature_wrapper_8528299Б !DвA
в 
:к7
5
dense_input&К#
dense_input         Д"1к.
,
dense_2!К
dense_2         
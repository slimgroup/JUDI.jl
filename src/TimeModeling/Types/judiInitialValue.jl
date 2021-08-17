############################################################
# judiInitialValue ##############################################
############################################################

# Authors: Rafael Orozco
# Date: April 2021


export judiInitialValue, judiInitialValueException

############################################################

# structure for initial value as an abstract vector
mutable struct judiInitialValue{vDT<:Number} <: joAbstractLinearOperator{vDT,vDT}
    name::String
    m::Integer
    n::Integer
    firstValue
    secondValue
end

mutable struct judiInitialValueException <: Exception
    msg :: String
end

############################################################

## outer constructors

"""
    judiInitialValue
        name::String
    	m::Integer
    	n::Integer
    	firstValue
    	secondValue
Abstract vector setting up the initial conditions for an Initial Value Problem. 
Constructors
============
Construct initialvalue structure:
    judiInitialValue(firstValue, secondValue)
"""
function judiInitialValue(firstValue::Array, secondValue::Array; vDT::DataType=Float32)
    (eltype(firstValue) != vDT) && (firstValue = convert(Array{vDT},firstValue))
    (eltype(secondValue) != vDT) && (secondValue = convert(Array{vDT},secondValue))
    # length of vector
    n = 1
    m = prod(size(firstValue))
    return judiInitialValue{vDT}("inital value source",m,n,firstValue,secondValue)
end


############################################################

# *(joLinearFunction, judiInitialValue)
function *(A::joLinearFunction{ADDT,ARDT},v::judiInitialValue{avDT}) where {ADDT, ARDT, avDT}
    A.n == size(v,1) || throw(judiInitialValueException("Shape mismatch: A:$((A.m, A.n)), v: $(size(v))"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiInitialValue):",A.name,typeof(A),avDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiInitialValue):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(joLinearOperator, judiInitialValue)
function *(A::joLinearOperator{ADDT,ARDT},v::judiInitialValue{avDT}) where {ADDT, ARDT, avDT}
    A.n == size(v,1) || throw(judiInitialValueException("Shape mismatch: A:$((A.m, A.n)), v: $(size(v))"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiInitialValue):",A.name,typeof(A),avDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiInitialValue):",A.name,typeof(A),eltype(V)]," / "))
    return V
end


# *(number, judiInitialValue)
function *(a::Number,v::judiInitialValue{avDT}) where {avDT}
    v.firstValue = v.firstValue*a
    v.secondValue = v.secondValue*a
    return v
end



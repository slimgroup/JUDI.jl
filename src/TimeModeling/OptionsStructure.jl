# Options structure 
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: May 2017
#

export Options

# Object for velocity/slowness models
type Options
    space_order::Integer
	retry_n::Integer
	limit_m::Bool
	buffer_size::Real
	save_data_to_disk::Bool
	file_path::String
	file_name::String
	sum_padding::Bool
    save_wavefield::Bool
end

"""
    Options
        space_order::Integer
        retry_n::Integer
        limit_m::Bool
        buffer_size::Real
        save_rate::Real
        save_data_to_disk::Bool
        file_path::String
        file_name::String
        sum_padding::Bool
		

Options structure for seismic modeling.

`space_order`: finite difference space order for wave equation (default is 8, needs to be multiple of 4)

`retry_n`: retry modeling operations in case of worker failure up to `retry_n` times

`limit_m`: for 3D modeling, limit modeling domain to area with receivers (saves memory)

`buffer_size`: if `limit_m=true`, define buffer area on each side of modeling domain (in meters)

`save_data_to_disk`: if `true`, saves shot records as separate SEG-Y files

`file_path`: path to directory where data is saved

`file_name`: shot records will be saved as specified file name plus its source coordinates

`sum_padding`: when removing the padding area of the gradient, sum into boundary rows/columns for true adjoints

`save_wavefield`: save forward wavefields and return as a second argument: (data, wavefield) = Pr*F*Ps'*q

Constructor
==========

All arguments are optional keyword arguments with the following default values:

    Options(;retry_n=0, limit_m=false, buffer_size=1e3, save_data_to_disk=false, file_path=pwd(), 
            file_name="shot", sum_padding=false, save_wavefield=false)

"""
Options(;space_order=8,retry_n=0,limit_m=false,buffer_size=1e3, save_data_to_disk=false, file_path="", file_name="shot", sum_padding=false, save_wavefield=false) = 
    Options(space_order,retry_n,limit_m,buffer_size,save_data_to_disk,file_path,file_name, sum_padding, save_wavefield)



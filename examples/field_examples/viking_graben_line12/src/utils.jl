export save_data, save_fhistory, rho_from_slowness


function save_data(x,z,data; pltfile,title,colormap,clim=nothing,h5file,h5openflag,h5varname)
  @info "save_data: $h5file"
  n = (length(x),length(z))
  o = (x[1],z[1])
  d = (x[2]-x[1],z[2]-z[1])
  isnothing(clim) && (clim = (minimum(data),maximum(data)))
  plt = Plots.heatmap(x, z, data, c=colormap, 
      xlims=(x[1],x[end]), 
      ylims=(z[1],z[end]), yflip=true,
      title=title,
      clim=clim,
      xlabel="Lateral position [km]",
      ylabel="Depth [km]",
      dpi=600)
  Plots.savefig(plt, pltfile)

  fid = h5open(h5file, h5openflag)
  (haskey(fid, h5varname)) && (delete_object(fid, h5varname))
  (haskey(fid, "o")) && (delete_object(fid, "o"))
  (haskey(fid, "n")) && (delete_object(fid, "n"))
  (haskey(fid, "d")) && (delete_object(fid, "d"))
  write(fid, 
      h5varname, Matrix(adjoint(data)), # convert adjoint(Matrix) type to Matrix
      "o", collect(o.*1000f0), 
      "n", collect(n), 
      "d", collect(d.*1000f0))
  close(fid)
end

function save_fhistory(fhistory; h5file,h5openflag,h5varname)
  @info "save_fhistory: $h5file"
  fid = h5open(h5file, h5openflag)
  (haskey(fid, h5varname)) && (delete_object(fid, h5varname))
  write(fid, h5varname, fhistory)
  close(fid)
end

"""
Recalculate slowness^2 to density using Gardner formulae
"""
rho_from_slowness(m) = 0.23.*(sqrt.(1f0 ./ m).*1000f0).^0.25
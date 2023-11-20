using DrWatson
@quickactivate "Viking"

using SegyIO

prestk_dir = "$(@__DIR__)/../"
prestk_file = "seismic.segy"
dt = 4                # ms
ns = 1000             # limit number of samples as there is no useful waves below 4 seconds
dir_out = "$(@__DIR__)/trim_segy/"

container = segy_scan(prestk_dir, prestk_file, ["SourceX", "SourceY", "GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])

# prepare folder for output data
mkpath(dir_out)

I = length(container)
progress = 0
for i in 1:I
  block = container[i]
  block_out = SeisBlock(Float32.(block.data[1:ns,:]))
  # without copying (copy is not supported for fileheader)
  block_out.fileheader = block.fileheader
  block_out.fileheader.bfh.DataSampleFormat = 5
  block_out.traceheaders = block.traceheaders
  block_out.fileheader.bfh.dt = Int(round(dt*1000f0))
  block_out.fileheader.bfh.ns = ns
  set_header!(block_out, "dt", Int(round(dt*1000f0)))
  set_header!(block_out, "ns", ns)
  segy_write(dir_out * "shot_$i.sgy", block_out)
  if round(i/I*100f0) > progress
    global progress = round(i/I*100f0)
    @info "progress: ($progress)%"
  end
end

# using Pkg
# Pkg.add("DSP")
# Pkg.add("SegyIO")
# Pkg.add("Suppressor")

using DSP, SegyIO, Suppressor

prestk_dir = "$(@__DIR__)/.."
prestk_file = "seismic.segy"
# frequencies may increase as follows: 0.005, 0.008, 0.012, 0.018, 0.025, 0.035
frq = 0.005           # kHz
dt = 4                # ms
ns = 1000             # limit number of samples as there is no useful waves below 4 seconds
dir_out = "$(@__DIR__)/filt/$(frq)hz_$(dt*ns)ms/"

container = segy_scan(prestk_dir, prestk_file, ["SourceX", "SourceY", "GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])

# prepare folder for output data
mkpath(dir_out)

# sampling frequency
fs = 1e3/dt
# Nyquist frequency
f_nyq = fs/2f0

responsetype = Lowpass(frq*1000f0; fs=fs)
designmethod = Butterworth(8)

I = length(container)
progress = 0
for i in 1:I
  block = container[i]
  block_out = SeisBlock(Float32.(filt(digitalfilter(responsetype, designmethod), Float32.(block.data[1:ns,:]))))
  # without copying
  block_out.fileheader = block.fileheader
  block_out.traceheaders = block.traceheaders
  block_out.fileheader.bfh.dt = Int(round(dt*1000f0))
  block_out.fileheader.bfh.ns = ns
  set_header!(block_out, "dt", Int(round(dt*1000f0)))
  set_header!(block_out, "ns", ns)
  @suppress_err segy_write(dir_out * "shot_$i.sgy", block_out)
  if round(i/I*100f0) > progress
    global progress = round(i/I*100f0)
    @info "progress: ($progress)%"
  end
end

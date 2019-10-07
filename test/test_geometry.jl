# Unit tests for JUDI Geometry structure
# Philipp Witte (pwitte.slim@gmail.com)
# May 2018
#

using JUDI.TimeModeling, SegyIO, Test, LinearAlgebra

@testset "Geometry Unit Test" begin

    # Number of sources
    nsrc = 2

    # Constructor if nt is not passed
    xsrc = convertToCell(range(100f0, stop=1100f0, length=nsrc))
    ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
    zsrc = convertToCell(range(20f0, stop=20f0, length=nsrc))

    geometry =  Geometry(xsrc, ysrc, zsrc; dt=2f0, t=1000f0)

    @test isequal(typeof(geometry.xloc), Array{Any, 1})
    @test isequal(typeof(geometry.yloc), Array{Any, 1})
    @test isequal(typeof(geometry.zloc), Array{Any, 1})
    @test isequal(typeof(geometry.nt), Array{Any, 1})
    @test isequal(typeof(geometry.dt), Array{Any, 1})
    @test isequal(typeof(geometry.t), Array{Any, 1})

    # Constructor if coordinates are not passed as a cell arrays
    xsrc = range(100f0, stop=1100f0, length=nsrc)
    ysrc = range(0f0, stop=0f0, length=nsrc)
    zsrc = range(20f0, stop=20f0, length=nsrc)

    geometry = Geometry(xsrc, ysrc, zsrc; dt=4f0, t=1000f0, nsrc=nsrc)

    @test isequal(typeof(geometry.xloc), Array{Any, 1})
    @test isequal(typeof(geometry.yloc), Array{Any, 1})
    @test isequal(typeof(geometry.zloc), Array{Any, 1})
    @test isequal(typeof(geometry.nt), Array{Any, 1})
    @test isequal(typeof(geometry.dt), Array{Any, 1})
    @test isequal(typeof(geometry.t), Array{Any, 1})

    # Set up source geometry object from in-core data container
    block = segy_read("../data/unit_test_shot_records.segy")
    src_geometry = Geometry(block; key="source", segy_depth_key="SourceSurfaceElevation")
    rec_geometry = Geometry(block; key="receiver", segy_depth_key="RecGroupElevation")

    @test isequal(typeof(src_geometry), GeometryIC)
    @test isequal(typeof(rec_geometry), GeometryIC)
    @test isequal(get_header(block, "SourceSurfaceElevation")[1], src_geometry.zloc[1])
    @test isequal(get_header(block, "RecGroupElevation")[1], rec_geometry.zloc[1][1])
    @test isequal(get_header(block, "SourceX")[1], src_geometry.xloc[1])
    @test isequal(get_header(block, "GroupX")[1], rec_geometry.xloc[1][1])

    # Set up geometry summary from out-of-core data container
    container = segy_scan("../data/", "unit_test_shot_records", ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
    src_geometry = Geometry(container; key="source", segy_depth_key="SourceSurfaceElevation")
    rec_geometry = Geometry(container; key="receiver", segy_depth_key="RecGroupElevation")

    @test isequal(typeof(src_geometry), GeometryOOC)
    @test isequal(typeof(rec_geometry), GeometryOOC)
    @test isequal(src_geometry.key, "source")
    @test isequal(rec_geometry.key, "receiver")
    @test isequal(src_geometry.segy_depth_key, "SourceSurfaceElevation")
    @test isequal(rec_geometry.segy_depth_key, "RecGroupElevation")
    @test isequal(prod(size(block.data)), sum(rec_geometry.nsamples))

    # Set up geometry summary from out-of-core data container passed as cell array
    container_cell = Array{SegyIO.SeisCon}(undef, nsrc)
    for j=1:nsrc
        container_cell[j] = split(container, j)
    end

    src_geometry = Geometry(container_cell; key="source", segy_depth_key="SourceSurfaceElevation")
    rec_geometry = Geometry(container_cell; key="receiver", segy_depth_key="RecGroupElevation")

    @test isequal(typeof(src_geometry), GeometryOOC)
    @test isequal(typeof(rec_geometry), GeometryOOC)
    @test isequal(src_geometry.key, "source")
    @test isequal(rec_geometry.key, "receiver")
    @test isequal(src_geometry.segy_depth_key, "SourceSurfaceElevation")
    @test isequal(rec_geometry.segy_depth_key, "RecGroupElevation")
    @test isequal(prod(size(block.data)), sum(rec_geometry.nsamples))

    # Load geometry from out-of-core Geometry container
    src_geometry_ic = Geometry(src_geometry)
    rec_geometry_ic = Geometry(rec_geometry)

    @test isequal(typeof(src_geometry_ic), GeometryIC)
    @test isequal(typeof(rec_geometry_ic), GeometryIC)
    @test isequal(get_header(block, "SourceSurfaceElevation")[1], src_geometry_ic.zloc[1])
    @test isequal(get_header(block, "RecGroupElevation")[1], rec_geometry_ic.zloc[1][1])
    @test isequal(get_header(block, "SourceX")[1], src_geometry_ic.xloc[1])
    @test isequal(get_header(block, "GroupX")[1], rec_geometry_ic.xloc[1][1])

    # Subsample in-core geometry structure
    src_geometry_sub = subsample(src_geometry_ic, 1)
    @test isequal(typeof(src_geometry_sub), GeometryIC)
    @test isequal(length(src_geometry_sub.xloc), 1)

    src_geometry_sub = subsample(src_geometry_ic, 1:2)
    @test isequal(typeof(src_geometry_sub), GeometryIC)
    @test isequal(length(src_geometry_sub.xloc), 2)

    # Subsample out-of-core geometry structure
    src_geometry_sub = subsample(src_geometry, 1)
    @test isequal(typeof(src_geometry_sub), GeometryOOC)
    @test isequal(length(src_geometry_sub.dt), 1)
    @test isequal(src_geometry_sub.segy_depth_key, "SourceSurfaceElevation")

    src_geometry_sub = subsample(src_geometry, 1:2)
    @test isequal(typeof(src_geometry_sub), GeometryOOC)
    @test isequal(length(src_geometry_sub.dt), 2)
    @test isequal(src_geometry_sub.segy_depth_key, "SourceSurfaceElevation")

    # Compare if geometries match
    @test compareGeometry(src_geometry_ic, src_geometry_ic)
    @test compareGeometry(rec_geometry_ic, rec_geometry_ic)

    @test compareGeometry(src_geometry, src_geometry)
    @test compareGeometry(rec_geometry, rec_geometry)

end

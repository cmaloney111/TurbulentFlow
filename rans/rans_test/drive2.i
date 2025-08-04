# 1 "/home/cmaloney111/Nek5000/core/drive2.f"
      subroutine initdim
C-------------------------------------------------------------------
C     
C     Transfer array dimensions to common
C     
C-------------------------------------------------------------------

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 8 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 8 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 9 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 9 "/home/cmaloney111/Nek5000/core/drive2.f"
C     
      NELT=LELT
      NELV=LELV
      
      NX1=LX1
      NY1=LY1
      NZ1=LZ1
      
      NX2=LX2
      NY2=LY2
      NZ2=LZ2
      
      NX3=LX3
      NY3=LY3
      NZ3=LZ3
      
      NXD=LXD
      NYD=LYD
      NZD=LZD
      
      NDIM=LDIM
C     
      RETURN
      END
C     
      subroutine initdat
C--------------------------------------------------------------------
C     
C     Initialize and set default values.
C     
C--------------------------------------------------------------------

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 41 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 41 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 42 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 42 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/CTIMER" 1
c     
c     Timer variables
c     
# 4
      real*8          tmxmf,tmxms,tdsum,taxhm,tcopy,tinvc,tinv3
     $               ,tinit,tadc3,tcol3,ta2s2,tcol2,tadd2
      common /ctimer/ tmxmf,tmxms,tdsum,taxhm,tcopy,tinvc,tinv3
     $               ,tinit,tadc3,tcol3,ta2s2,tcol2,tadd2
c     
      real*8          tsolv,tgsum,tdsnd,tdadd,tcdtp,tmltd,tprep
     $               ,tpres,thmhz,tgop ,tgop1,tdott,tbsol,tbso2
     $               ,tsett,tslvb,tusbc,tddsl,tcrsl,tdsmx,tdsmn
     $               ,tgsmn,tgsmx,teslv,tbbbb,tcccc,tdddd,teeee
     $               ,tvdss,tschw,tadvc,tspro,tgop_sync,tsyc
     $               ,twal,tgp2,tcvf,tproj,tusfq,tuchk
     $               ,tmakf,tmakq
      common /ctime2/ tsolv,tgsum,tdsnd,tdadd,tcdtp,tmltd,tprep
     $               ,tpres,thmhz,tgop ,tgop1,tdott,tbsol,tbso2
     $               ,tsett,tslvb,tusbc,tddsl,tcrsl,tdsmx,tdsmn
     $               ,tgsmn,tgsmx,teslv,tbbbb,tcccc,tdddd,teeee
     $               ,tvdss,tschw,tadvc,tspro,tgop_sync,tsyc
     $               ,twal,tgp2,tcvf,tproj,tusfq,tuchk
     $               ,tmakf,tmakq
c     
      integer nmxmf,nmxms,ndsum,naxhm,ncopy,ninvc,ninv3
      common /itimer/ nmxmf,nmxms,ndsum,naxhm,ncopy,ninvc,ninv3
c     
      integer         nsolv,ngsum,ndsnd,ndadd,ncdtp,nmltd,nprep
     $               ,npres,nhmhz,ngop ,ngop1,ndott,nbsol,nbso2
     $               ,nsett,nslvb,nusbc,nddsl,ncrsl,ndsmx,ndsmn
     $               ,ngsmn,ngsmx,neslv,nbbbb,ncccc,ndddd,neeee
     $               ,nvdss,nadvc,nspro,ngop_sync,nsyc,nwal,ngp2
     $               ,ncvf
      common /itime2/ nsolv,ngsum,ndsnd,ndadd,ncdtp,nmltd,nprep
     $               ,npres,nhmhz,ngop ,ngop1,ndott,nbsol,nbso2
     $               ,nsett,nslvb,nusbc,nddsl,ncrsl,ndsmx,ndsmn
     $               ,ngsmn,ngsmx,neslv,nbbbb,ncccc,ndddd,neeee
     $               ,nvdss,nadvc,nspro,ngop_sync,nsyc,nwal,ngp2
     $               ,ncvf
c     
      real*8          pmxmf,pmxms,pdsum,paxhm,pcopy,pinvc,pinv3
     $               ,psolv,pgsum,pdsnd,pdadd,pcdtp,pmltd,pprep
     $               ,ppres,phmhz,pgop ,pgop1,pdott,pbsol,pbso2
     $               ,psett,pslvb,pusbc,pddsl,pcrsl,pdsmx,pdsmn
     $               ,pgsmn,pgsmx,peslv,pbbbb,pcccc,pdddd,peeee
     $               ,pvdss,pspro,pgop_sync,psyc,pwal,pgp2
      common /ptimer/ pmxmf,pmxms,pdsum,paxhm,pcopy,pinvc,pinv3
     $               ,psolv,pgsum,pdsnd,pdadd,pcdtp,pmltd,pprep
     $               ,ppres,phmhz,pgop ,pgop1,pdott,pbsol,pbso2
     $               ,psett,pslvb,pusbc,pddsl,pcrsl,pdsmx,pdsmn
     $               ,pgsmn,pgsmx,peslv,pbbbb,pcccc,pdddd,peeee
     $               ,pvdss,pspro,pgop_sync,psyc,pwal,pgp2
      
c     
      real*8 etime1,etime2,etime0,gtime1,tscrtch
      real*8 dnekclock,dnekclock_sync
c     
      real*8          etimes,ttotal,tttstp,etims0,ttime
      common /ctime3/ etimes,ttotal,tttstp,etims0,ttime
c     
      integer icalld
      save    icalld
      data    icalld /0/
      
      logical         ifsync
      common /ctimel/ ifsync
# 43 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 43 "/home/cmaloney111/Nek5000/core/drive2.f"
      COMMON /DOIT/ IFDOIT
      LOGICAL       IFDOIT
      
c     Set default logicals
      
      ifdoit    = .false.
      ifcvode   = .false.
      ifexplvis = .false.
      ifvvisp   = .true.
      
      ifsplit = .false.
      if (lx1.eq.lx2) ifsplit=.true.
      
      if_full_pres = .false.
      
      CALL RZERO (PARAM,200)
      
      CALL BLANK(CCURVE ,12*LELT)
      NEL8 = 8*LELT
      CALL RZERO(XC,NEL8)
      CALL RZERO(YC,NEL8)
      CALL RZERO(ZC,NEL8)
      
      NTOT=lx1*ly1*lz1*LELT
      CALL RZERO(ABX1,NTOT)
      CALL RZERO(ABX2,NTOT)
      CALL RZERO(ABY1,NTOT)
      CALL RZERO(ABY2,NTOT)
      CALL RZERO(ABZ1,NTOT)
      CALL RZERO(ABZ2,NTOT)
      CALL RZERO(VGRADT1,NTOT)
      CALL RZERO(VGRADT2,NTOT)
      
      NTOT=lx2*ly2*lz2*LELT
      CALL RZERO(USRDIV,NTOT)
      CALL RZERO(QTL,NTOT)
      
      CALL IONE(out_mask, lelt)
      
      NSTEPS = 0
      
      RETURN
      END
C     
      subroutine comment
C---------------------------------------------------------------------
C     
C     No need to comment !!
C     
C---------------------------------------------------------------------

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 94 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 94 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 95 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 95 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 96 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 96 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 97 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 97 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/CTIMER" 1
c     
c     Timer variables
c     
# 4
      real*8          tmxmf,tmxms,tdsum,taxhm,tcopy,tinvc,tinv3
     $               ,tinit,tadc3,tcol3,ta2s2,tcol2,tadd2
      common /ctimer/ tmxmf,tmxms,tdsum,taxhm,tcopy,tinvc,tinv3
     $               ,tinit,tadc3,tcol3,ta2s2,tcol2,tadd2
c     
      real*8          tsolv,tgsum,tdsnd,tdadd,tcdtp,tmltd,tprep
     $               ,tpres,thmhz,tgop ,tgop1,tdott,tbsol,tbso2
     $               ,tsett,tslvb,tusbc,tddsl,tcrsl,tdsmx,tdsmn
     $               ,tgsmn,tgsmx,teslv,tbbbb,tcccc,tdddd,teeee
     $               ,tvdss,tschw,tadvc,tspro,tgop_sync,tsyc
     $               ,twal,tgp2,tcvf,tproj,tusfq,tuchk
     $               ,tmakf,tmakq
      common /ctime2/ tsolv,tgsum,tdsnd,tdadd,tcdtp,tmltd,tprep
     $               ,tpres,thmhz,tgop ,tgop1,tdott,tbsol,tbso2
     $               ,tsett,tslvb,tusbc,tddsl,tcrsl,tdsmx,tdsmn
     $               ,tgsmn,tgsmx,teslv,tbbbb,tcccc,tdddd,teeee
     $               ,tvdss,tschw,tadvc,tspro,tgop_sync,tsyc
     $               ,twal,tgp2,tcvf,tproj,tusfq,tuchk
     $               ,tmakf,tmakq
c     
      integer nmxmf,nmxms,ndsum,naxhm,ncopy,ninvc,ninv3
      common /itimer/ nmxmf,nmxms,ndsum,naxhm,ncopy,ninvc,ninv3
c     
      integer         nsolv,ngsum,ndsnd,ndadd,ncdtp,nmltd,nprep
     $               ,npres,nhmhz,ngop ,ngop1,ndott,nbsol,nbso2
     $               ,nsett,nslvb,nusbc,nddsl,ncrsl,ndsmx,ndsmn
     $               ,ngsmn,ngsmx,neslv,nbbbb,ncccc,ndddd,neeee
     $               ,nvdss,nadvc,nspro,ngop_sync,nsyc,nwal,ngp2
     $               ,ncvf
      common /itime2/ nsolv,ngsum,ndsnd,ndadd,ncdtp,nmltd,nprep
     $               ,npres,nhmhz,ngop ,ngop1,ndott,nbsol,nbso2
     $               ,nsett,nslvb,nusbc,nddsl,ncrsl,ndsmx,ndsmn
     $               ,ngsmn,ngsmx,neslv,nbbbb,ncccc,ndddd,neeee
     $               ,nvdss,nadvc,nspro,ngop_sync,nsyc,nwal,ngp2
     $               ,ncvf
c     
      real*8          pmxmf,pmxms,pdsum,paxhm,pcopy,pinvc,pinv3
     $               ,psolv,pgsum,pdsnd,pdadd,pcdtp,pmltd,pprep
     $               ,ppres,phmhz,pgop ,pgop1,pdott,pbsol,pbso2
     $               ,psett,pslvb,pusbc,pddsl,pcrsl,pdsmx,pdsmn
     $               ,pgsmn,pgsmx,peslv,pbbbb,pcccc,pdddd,peeee
     $               ,pvdss,pspro,pgop_sync,psyc,pwal,pgp2
      common /ptimer/ pmxmf,pmxms,pdsum,paxhm,pcopy,pinvc,pinv3
     $               ,psolv,pgsum,pdsnd,pdadd,pcdtp,pmltd,pprep
     $               ,ppres,phmhz,pgop ,pgop1,pdott,pbsol,pbso2
     $               ,psett,pslvb,pusbc,pddsl,pcrsl,pdsmx,pdsmn
     $               ,pgsmn,pgsmx,peslv,pbbbb,pcccc,pdddd,peeee
     $               ,pvdss,pspro,pgop_sync,psyc,pwal,pgp2
      
c     
      real*8 etime1,etime2,etime0,gtime1,tscrtch
      real*8 dnekclock,dnekclock_sync
c     
      real*8          etimes,ttotal,tttstp,etims0,ttime
      common /ctime3/ etimes,ttotal,tttstp,etims0,ttime
c     
      integer icalld
      save    icalld
      data    icalld /0/
      
      logical         ifsync
      common /ctimel/ ifsync
# 98 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 98 "/home/cmaloney111/Nek5000/core/drive2.f"
      
      LOGICAL  IFCOUR
      SAVE     IFCOUR
      COMMON  /CPRINT/ IFPRINT
      LOGICAL          IFPRINT
      REAL*8 EETIME0,EETIME1,EETIME2
      SAVE   EETIME0,EETIME1,EETIME2
      DATA   EETIME0,EETIME1,EETIME2 /0.0, 0.0, 0.0/
      
C     
C     Only node zero makes comments.
      IF (NIO.NE.0) RETURN
C     
C     
      IF (EETIME0.EQ.0.0 .AND. ISTEP.EQ.1) EETIME0=DNEKCLOCK()
      EETIME1=EETIME2
      EETIME2=DNEKCLOCK()
C     
      IF (ISTEP.EQ.0) THEN
         IFCOUR  = .FALSE.
         DO 10 IFIELD=1,NFIELD
            IF (IFADVC(IFIELD)) IFCOUR = .TRUE.
 10      CONTINUE
         IF (IFWCNO) IFCOUR = .TRUE.
      ELSEIF (ISTEP.GT.0 .AND. LASTEP.EQ.0 .AND. IFTRAN) THEN
         TTIME_STP = EETIME2-EETIME1   ! time per timestep
         TTIME     = EETIME2-EETIME0   ! sum of all timesteps
         IF(ISTEP.EQ.1) THEN
           TTIME_STP = 0
           TTIME     = 0
         ENDIF
         IF (     IFCOUR) 
     $       WRITE(6,100)ISTEP,TIME,DT,COURNO,TTIME,TTIME_STP
         IF (.NOT.IFCOUR) WRITE (6,101) ISTEP,TIME,DT
      ELSEIF (LASTEP.EQ.1) THEN
         TTIME_STP = EETIME2-EETIME1   ! time per timestep
         TTIME     = EETIME2-EETIME0   ! sum of all timesteps
      ENDIF
 100  FORMAT('Step',I7,', t=',1pE14.7,', DT=',1pE14.7
     $,', C=',0pF7.3,2(1pE11.4))
 101  FORMAT('Step',I7,', time=',1pE12.5,', DT=',1pE11.3)
      
      RETURN
      END
C     
      subroutine setvar
C-----------------------------------------------------------------------
C     
C     Initialize variables
C     
C-----------------------------------------------------------------------

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 150 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 150 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 151 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 151 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 152 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 152 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 153 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 153 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 154 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 154 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/NEKNEK" 1
c     
c     Multimesh variables
c     
# 4
      integer intflag(2*ldim,lelt)
      common /intflag/ intflag
      
      integer imask(lx1,ly1,lz1,lelt)
      common /intmask/ imask 
      
      real             valint(lx1,ly1,lz1,lelt,nfldmax_nn)
      common /valmask/ valint
      
      integer igeom
      common /cgeom/ igeom
      
      integer nfld_neknek
      common /inbc/ nfld_neknek
      
      real bdrylg(lx1*ly1*lz1*lelt,nfldmax_nn,0:2)
      common /mybd/ bdrylg
      
      real    rst(nmaxl_nn*ldim)
      common /multipts_r/ rst
      
      integer rcode(nmaxl_nn),elid(nmaxl_nn)
     $    ,proc(nmaxl_nn),ilist(1,nmaxl_nn),npoints_nn
      common /multipts_i/ rcode,elid,proc,ilist,npoints_nn
      
      integer inth_multi2
      common /intp_h_nn/ inth_multi2
# 155 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 155 "/home/cmaloney111/Nek5000/core/drive2.f"
      
      param(120) = 500 ! print runtime stats
      
C     
C     Geometry on Mesh 3 or 1?
C     
      IFGMSH3 = .TRUE.
      IF ( IFSTRS )           IFGMSH3 = .FALSE.
      IF (.NOT.IFFLOW)        IFGMSH3 = .FALSE.
      IF ( IFSPLIT )          IFGMSH3 = .FALSE.
      
      NGEOM  = 2
C     
      NFIELD = 1
      IF (IFHEAT) THEN
         NFIELD = 2 + NPSCAL
         NFLDTM = 1 + NPSCAL
      ENDIF
c     
      nfldt = nfield
      if (ifmhd) then
         nfldt  = nfield + 1
         nfldtm = nfldtm + 1
      endif
c     
      MFIELD = 1
      IF (IFMVBD) MFIELD = 0
C     
      DO 100 IFIELD=MFIELD,nfldt+(LDIMT-1 - NPSCAL)
         IF (IFTMSH(IFIELD)) THEN
             NELFLD(IFIELD) = NELT
         ELSE
             NELFLD(IFIELD) = NELV
         ENDIF
 100  CONTINUE
      
      ! Maximum iteration counts for linear solver
      NMXV   = 1000
      if (iftran) NMXV = 200
      NMXH   =  NMXV ! not used anymore
      NMXP   = 200
      do ifield = 2,ldimt+1
         NMXT(ifield-1) = 200 
      enddo 
      NMXE   = 100
      NMXNL  = 10 
C     
      PARAM(86) = 0 ! No skew-symm. convection for now
C     
      DT     = abs(PARAM(12))
      DTINIT = DT
      FINTIM = PARAM(10)
      NSTEPS = PARAM(11)
      IOCOMM = PARAM(13)
      TIMEIO = PARAM(14)
      IOSTEP = PARAM(15)
      LASTEP = 0
      TOLPDF = abs(PARAM(21))
      TOLHDF = abs(PARAM(22))
      TOLREL = abs(PARAM(24))
      TOLABS = abs(PARAM(25))
      CTARG  = PARAM(26)
      NBDINP = abs(PARAM(27))
      NABMSH = PARAM(28)
      
      if (nbdinp.gt.lorder) then
         if (nid.eq.0) then
           write(6,*) 'ERROR: torder > lorder.',nbdinp,lorder
           write(6,*) 'Change SIZE and recompile entire code.'
         endif
         call exitt
      endif
      
C     Check accuracy requested.
C     
      IF (TOLREL.LE.0.) TOLREL = 0.01
C     
C     Relaxed pressure iteration; maximum decrease in the residual.
C     
      PRELAX = 0.1*TOLREL
      IF (.NOT.IFTRAN .AND. .NOT.IFNAV) PRELAX = 1.E-5
C     
C     Tolerance for nonlinear iteration
C     
      TOLNL  = 1.E-4
C     
C     Fintim overrides nsteps
C     
      IF (FINTIM.NE.0.) NSTEPS = 1000000000
      IF (.NOT.IFTRAN ) NSTEPS = 1
C     
C     Print interval defaults to 1
C     
      IF (IOCOMM.EQ.0)  IOCOMM = nsteps+1
C     
C     Set default for mesh integration scheme
C     
      IF (NABMSH.LE.0 .OR. NABMSH.GT.3) THEN
         NABMSH    = NBDINP
         PARAM(28) = (NABMSH)
      ENDIF
C     
C     Courant number only applicable if convection in ANY field.
C     
      IADV  = 0
      IFLD1 = 1
      IF (.NOT.IFFLOW) IFLD1 = 2
      DO 200 IFIELD=IFLD1,nfldt
         IF (IFADVC(IFIELD)) IADV = 1
 200  CONTINUE
C     
C     If characteristics, need number of sub-timesteps (DT/DS).
C     Current sub-timeintegration scheme: RK4.
C     If not characteristics, i.e. standard semi-implicit scheme,
C     check user-defined Courant number.
C     
      IF (IADV.EQ.1) CALL SETCHAR
C     
C     Initialize order of time-stepping scheme (BD)
C     Initialize time step array.
C     
      NBD    = 0
      CALL RZERO (DTLAG,10)
      
      ! neknek 
      ifneknekm = .false.
      ninter = 1
      nfld_neknek = ndim + nfield
      
      CALL BLANK(cbc_bmap,sizeof(cbc_bmap))
      
      one = 1.
      PI  = 4.*ATAN(one)
      
      RETURN
      END
C     
      subroutine echopar
C     
C     Echo the nonzero parameters from the readfile to the logfile
C     

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 297 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 297 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 298 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 298 "/home/cmaloney111/Nek5000/core/drive2.f"
      CHARACTER*132 STRING
      CHARACTER*1  STRING1(132)
      EQUIVALENCE (STRING,STRING1)
C     
      IF (nid.ne.0) RETURN
C     
      OPEN (UNIT=9,FILE=REAFLE,STATUS='OLD')
      REWIND(UNIT=9)
C     
C     
      READ(9,*,ERR=400)
      READ(9,*,ERR=400) VNEKTON
      NKTONV=VNEKTON
      VNEKMIN=2.5
      IF(VNEKTON.LT.VNEKMIN)THEN
         PRINT*,' Error: This NEKTON Solver Requires a .rea file'
         PRINT*,' from prenek version ',VNEKMIN,' or higher'
         PRINT*,' Please run the session through the preprocessor'
         PRINT*,' to bring the .rea file up to date.'
         call exitt
      ENDIF
      READ(9,*,ERR=400) ldimr
c     error check
      IF(ldimr.NE.LDIM)THEN
         WRITE(6,10) LDIMR,ldim
   10       FORMAT(//,2X,'Error: This NEKTON Solver has been compiled'
     $              /,2X,'       for spatial dimension equal to',I2,'.'
     $              /,2X,'       The data file has dimension',I2,'.')
         CALL exitt
      ENDIF
C     
      CALL BLANK(STRING,132)
c      CALL CHCOPY(STRING,REAFLE,132)
      Ls=LTRUNC(STRING,132)
      READ(9,*,ERR=400) NPARAM
      WRITE(6,82) NPARAM,(STRING1(j),j=1,Ls)
C     
      DO 20 I=1,NPARAM
         CALL BLANK(STRING,132)
         READ(9,80,ERR=400) STRING
         Ls=LTRUNC(STRING,132)
         IF (PARAM(i).ne.0.0) WRITE(6,81) I,(STRING1(j),j=1,Ls)
   20 CONTINUE
   80 FORMAT(A132) 
   81 FORMAT(I4,3X,132A1)
   82 FORMAT(I4,3X,'Parameters from file:',132A1)
      CLOSE (UNIT=9)
      write(6,*) ' '
      
c      if(param(2).ne.param(8).and.nio.eq.0) then
c         write(6,*) 'Note VISCOS not equal to CONDUCT!'
c         write(6,*) 'Note VISCOS  =',PARAM(2)
c         write(6,*) 'Note CONDUCT =',PARAM(8)
c      endif
c     
      return
C     
C     Error handling:
C     
  400 CONTINUE
      WRITE(6,401)
  401 FORMAT(2X,'ERROR READING PARAMETER DATA'
     $    ,/,2X,'ABORTING IN ROUTINE ECHOPAR.')
      CALL exitt
C     
  500 CONTINUE
      WRITE(6,501)
  501 FORMAT(2X,'ERROR READING LOGICAL DATA'
     $    ,/,2X,'ABORTING IN ROUTINE ECHOPAR.')
      CALL exitt
C     
      RETURN
      END
C     
      subroutine gengeom (igeom)
C----------------------------------------------------------------------
C     
C     Generate geometry data
C     
C----------------------------------------------------------------------

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 379 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 379 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 380 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 380 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 381 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 381 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 382 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 382 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 383 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 383 "/home/cmaloney111/Nek5000/core/drive2.f"
C     
      COMMON /SCRUZ/ XM3 (LX3,LY3,LZ3,LELT)
     $ ,             YM3 (LX3,LY3,LZ3,LELT)
     $ ,             ZM3 (LX3,LY3,LZ3,LELT)
C     
      
      if (nio.eq.0.and.istep.le.1) write(6,*) 'generate geometry data'
      
      IF (IGEOM.EQ.1) THEN
         RETURN
      ELSEIF (IGEOM.EQ.2) THEN
         CALL LAGMASS
         IF (ISTEP.EQ.0) CALL GENCOOR (XM3,YM3,ZM3)
         IF (ISTEP.GE.1) CALL UPDCOOR
         CALL GEOM1 (XM3,YM3,ZM3)
         CALL GEOM2
         CALL UPDMSYS (1)
         CALL VOLUME
         CALL SETINVM
         CALL SETDEF
         CALL SFASTAX
      ENDIF
      
      if (nio.eq.0.and.istep.le.1) then
        write(6,*) 'done :: generate geometry data' 
        write(6,*) ' '
      endif
      
      return
      end
c-----------------------------------------------------------------------
      subroutine files
C     
C     Defines machine specific input and output file names.
C     

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 419 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 419 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 420 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 420 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 421 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 421 "/home/cmaloney111/Nek5000/core/drive2.f"
C     
      CHARACTER*132 NAME
      CHARACTER*1   SESS1(132),PATH1(132),NAM1(132)
      EQUIVALENCE  (SESSION,SESS1)
      EQUIVALENCE  (PATH,PATH1)
      EQUIVALENCE  (NAME,NAM1)
      CHARACTER*1  DMP(4),FLD(4),REA(4),HIS(4),SCH(4) ,ORE(4), NRE(4)
      CHARACTER*1  RE2(4),PAR(4)
      CHARACTER*4  DMP4  ,FLD4  ,REA4  ,HIS4  ,SCH4   ,ORE4  , NRE4
      CHARACTER*4  RE24  ,PAR4
      EQUIVALENCE (DMP,DMP4), (FLD,FLD4), (REA,REA4), (HIS,HIS4)
     $          , (SCH,SCH4), (ORE,ORE4), (NRE,NRE4)
     $          , (RE2,RE24), (PAR,PAR4)
      DATA DMP4,FLD4,REA4 /'.dmp','.fld','.rea'/
      DATA HIS4,SCH4      /'.his','.sch'/
      DATA ORE4,NRE4      /'.ore','.nre'/
      DATA RE24           /'.re2'       /
      DATA PAR4           /'.par'       /
      CHARACTER*78  STRING
C     
C     Find out the session name:
C     
c      CALL BLANK(SESSION,132)
c      CALL BLANK(PATH   ,132)
      
c      ierr = 0
c      IF(NID.EQ.0) THEN
c        OPEN (UNIT=8,FILE='SESSION.NAME',STATUS='OLD',ERR=24)
c        READ(8,10) SESSION
c        READ(8,10) PATH
c  10      FORMAT(A132)
c        CLOSE(UNIT=8)
c        GOTO 23
c  24    ierr = 1
c  23  ENDIF
c      call err_chk(ierr,' Cannot open SESSION.NAME!$')
      
c      len = ltrunc(path,132)
c      if(len.lt.1) then
c         call chcopy(path1(1),'./',2)
c      endif
      
c      call bcast(SESSION,132*CSIZE)
c      call bcast(PATH,132*CSIZE)
      
      CALL BLANK(PARFLE,132)
      CALL BLANK(REAFLE,132)
      CALL BLANK(RE2FLE,132)
      CALL BLANK(FLDFLE,132)
      CALL BLANK(HISFLE,132)
      CALL BLANK(SCHFLE,132)
      CALL BLANK(DMPFLE,132)
      CALL BLANK(OREFLE,132)
      CALL BLANK(NREFLE,132)
      CALL BLANK(NAME  ,132)
C     
C     Construct file names containing full path to host:
C     
      LS=LTRUNC(SESSION,132)
      LPP=0 !LTRUNC(PATH,132)
      LSP=LS+LPP
c     
      call chcopy(nam1(    1),path1,lpp)
      call chcopy(nam1(lpp+1),sess1,ls )
      l1 = lpp+ls+1
      ln = lpp+ls+4
c     
c     
c .rea file
      call chcopy(nam1  (l1),rea , 4)
      call chcopy(reafle    ,nam1,ln)
c      write(6,*) 'reafile:',reafle
c     
c .par file
      call chcopy(nam1  (l1),par , 4)
      call chcopy(parfle    ,nam1,ln)
c     
c .re2 file
      call chcopy(nam1  (l1),re2 , 4)
      call chcopy(re2fle    ,nam1,ln)
c     
c .fld file
      call chcopy(nam1  (l1),fld , 4)
      call chcopy(fldfle    ,nam1,ln)
c     
c .his file
      call chcopy(nam1  (l1),his , 4)
      call chcopy(hisfle    ,nam1,ln)
c     
c .sch file
      call chcopy(nam1  (l1),sch , 4)
      call chcopy(schfle    ,nam1,ln)
c     
c     
c .dmp file
      call chcopy(nam1  (l1),dmp , 4)
      call chcopy(dmpfle    ,nam1,ln)
c     
c .ore file
      call chcopy(nam1  (l1),ore , 4)
      call chcopy(orefle    ,nam1,ln)
c     
c .nre file
      call chcopy(nam1  (l1),nre , 4)
      call chcopy(nrefle    ,nam1,ln)
c     
C     Write the name of the .rea file to the logfile.
C     
C      IF (NIO.EQ.0) THEN
C         CALL CHCOPY(STRING,REAFLE,78)
C         WRITE(6,1000) STRING
C         WRITE(6,1001) 
C 1000    FORMAT(//,1X,'Beginning session:',/,2X,A78)
C 1001    FORMAT(/,' ')
C      ENDIF
C     
      RETURN
      
      END
C     
      subroutine settime
C----------------------------------------------------------------------
C     
C     Store old time steps and compute new time step, time and timef.
C     Set time-dependent coefficients in time-stepping schemes.
C     
C----------------------------------------------------------------------

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 549 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 549 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 550 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 550 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 551 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 551 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 552 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 552 "/home/cmaloney111/Nek5000/core/drive2.f"
      COMMON  /CPRINT/ IFPRINT
      LOGICAL          IFPRINT
      SAVE
C     
      irst = param(46)
C     
C     Set time step.
C     
      DO 10 ILAG=10,2,-1
         DTLAG(ILAG) = DTLAG(ILAG-1)
 10   CONTINUE
      CALL SETDT
      DTLAG(1) = DT
      IF (ISTEP.EQ.1 .and. irst.le.0) DTLAG(2) = DT
C     
C     Set time.
C     
      TIMEF    = TIME
      TIME     = TIME+DT
C     
C     Set coefficients in AB/BD-schemes.
C     
      CALL SETORDBD
      if (irst.gt.0) nbd = nbdinp
      CALL RZERO (BD,10)
      CALL SETBD (BD,DTLAG,NBD)
      if (PARAM(27).lt.0) then
         NAB = NBDINP
      else
         NAB = 3
      endif
      IF (ISTEP.lt.NAB.and.irst.le.0) NAB = ISTEP
      CALL RZERO   (AB,10)
      CALL SETABBD (AB,DTLAG,NAB,NBD)
      IF (IFMVBD) THEN
         NBDMSH = 1
         NABMSH = PARAM(28)
         IF (NABMSH.GT.ISTEP .and. irst.le.0) NABMSH = ISTEP
         IF (IFSURT)          NABMSH = NBD
         CALL RZERO   (ABMSH,10)
         CALL SETABBD (ABMSH,DTLAG,NABMSH,NBDMSH)
      ENDIF
      
C     
C     Set logical for printout to screen/log-file
C     
      IFPRINT = .FALSE.
      IF (IOCOMM.GT.0.AND.MOD(ISTEP,IOCOMM).EQ.0) IFPRINT=.TRUE.
      IF (ISTEP.eq.1  .or. ISTEP.eq.0           ) IFPRINT=.TRUE.
      IF (NIO.eq.-1)  ifprint=.false.
      
      RETURN
      END
      
      
      subroutine geneig (igeom)
C-----------------------------------------------------------------------
C     
C     Compute eigenvalues. 
C     Used for automatic setting of tolerances and to find critical
C     time step for explicit mode. 
C     Currently eigenvalues are computed only for the velocity mesh.
C     
C-----------------------------------------------------------------------

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 617 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 617 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 618 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 618 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 619 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 619 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 620 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 620 "/home/cmaloney111/Nek5000/core/drive2.f"
C     
      IF (IGEOM.EQ.1) RETURN
C     
C     Decide which eigenvalues to be computed.
C     
      IF (IFFLOW) THEN
C     
         IFAA  = .FALSE.
         IFAE  = .FALSE.
         IFAS  = .FALSE.
         IFAST = .FALSE.
         IFGA  = .TRUE.
         IFGE  = .FALSE.
         IFGS  = .FALSE.
         IFGST = .FALSE.
C     
C        For now, only compute eigenvalues during initialization.
C        For deforming geometries the eigenvalues should be 
C        computed every time step (based on old eigenvectors => more mem
C     
         IMESH  = 1
         IFIELD = 1
         TOLEV  = 1.E-3
         TOLHE  = TOLHDF
         TOLHR  = TOLHDF
         TOLHS  = TOLHDF
         TOLPS  = TOLPDF
         CALL EIGENV
         CALL ESTEIG
C     
      ELSEIF (IFHEAT.AND..NOT.IFFLOW) THEN
C     
         CALL ESTEIG
C     
      ENDIF
C     
      RETURN
      END
C-----------------------------------------------------------------------
      subroutine fluid (igeom)
C     
C     Driver for solving the incompressible Navier-Stokes equations.
C     
C     Current version:
C     (1) Velocity/stress formulation.
C     (2) Constant/variable properties.
C     (3) Implicit/explicit time stepping.
C     (4) Automatic setting of tolerances .
C     (5) Lagrangian/"Eulerian"(operator splitting) modes
C     
C-----------------------------------------------------------------------

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 672 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 672 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 673 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 673 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 674 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 674 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 675 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 675 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 676 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 676 "/home/cmaloney111/Nek5000/core/drive2.f"
      
      real*8 ts, dnekclock
      
      ifield = 1
      imesh  = 1
      call unorm
      call settolv
      
      ts = dnekclock() 
      
      if(nio.eq.0 .and. igeom.eq.2) 
     &   write(*,'(13x,a)') 'Solving for fluid'
      
      if (ifsplit) then
      
c        PLAN 4: TOMBO SPLITTING
c                - Time-dependent Navier-Stokes calculation (Re>>1).
c                - Same approximation spaces for pressure and velocity.
c                - Incompressibe or Weakly compressible (div u .ne. 0).
      
         call plan4 (igeom)                                           
         if (igeom.ge.2) call chkptol         ! check pressure tolerance
         if (igeom.eq.ngeom) then
           if (ifneknekc) then
              call vol_flow_ms    ! check for fixed flow rate
           else
              call vol_flow       ! check for fixed flow rate
           endif
         endif
         if (igeom.ge.2) call printdiverr
      
      elseif (iftran) then
      
c        call plan1 (igeom)       !  Orig. NEKTON time stepper
      
         if (ifrich) then
            call plan5(igeom)
         else
            call plan3 (igeom)    !  Same as PLAN 1 w/o nested iteration
                                  !  Std. NEKTON time stepper  !
         endif
      
         if (igeom.ge.2) call chkptol         ! check pressure tolerance
         if (igeom.eq.ngeom) then 
           if (ifneknekc) then
              call vol_flow_ms    ! check for fixed flow rate
           else
              call vol_flow       ! check for fixed flow rate
           endif
         endif
      
      else   !  steady Stokes, non-split
      
c             - Steady/Unsteady Stokes/Navier-Stokes calculation.
c             - Consistent approximation spaces for velocity and pressur
c             - Explicit treatment of the convection term. 
c             - Velocity/stress formulation.
      
         call plan1 (igeom) ! The NEKTON "Classic".
      
      endif
      
      if(nio.eq.0 .and. igeom.ge.2) 
     &   write(*,'(4x,i7,a,1p2e12.4)') 
     &   istep,'  Fluid done',time,dnekclock()-ts
      
      return
      end
c-----------------------------------------------------------------------
      subroutine heat (igeom)
C     
C     Driver for temperature or passive scalar.
C     
C     Current version:
C     (1) Varaiable properties.
C     (2) Implicit time stepping.
C     (3) User specified tolerance for the Helmholtz solver
C         (not based on eigenvalues).
C     (4) A passive scalar can be defined on either the 
C         temperatur or the velocity mesh.
C     (5) A passive scalar has its own multiplicity (B.C.).  
C     

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 759 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 759 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 760 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 760 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 761 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 761 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 762 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 762 "/home/cmaloney111/Nek5000/core/drive2.f"
      
      real*8 ts, dnekclock
      
      ts = dnekclock()
      
      if (nio.eq.0 .and. igeom.eq.2) 
     &    write(*,'(13x,a)') 'Solving for Hmholtz scalars'
      
      do ifield = 2,nfield
         if (idpss(ifield-1).eq.0) then      ! helmholtz
            intype        = -1
            if (.not.iftmsh(ifield)) imesh = 1
            if (     iftmsh(ifield)) imesh = 2
            call unorm
            call settolt
            call cdscal(igeom)
         endif
      enddo
      
      if (nio.eq.0 .and. igeom.eq.2)
     &   write(*,'(4x,i7,a,1p2e12.4)') 
     &   istep,'  Scalars done',time,dnekclock()-ts
      
      return
      end
c-----------------------------------------------------------------------
      subroutine heat_cvode (igeom)
C     

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 791 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 791 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 792 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 792 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 793 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 793 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 794 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 794 "/home/cmaloney111/Nek5000/core/drive2.f"
      
      real*8 ts, dnekclock
      
      ts = dnekclock()
      
      if (igeom.ne.2) return
      
      if (nio.eq.0) 
     &    write(*,'(13x,a)') 'Solving for CVODE scalars'
      
      call cdscal_cvode(igeom)
      
      if (nio.eq.0)
     &   write(*,'(4x,i7,a,1p2e12.4)') 
     &   istep,'  CVODE scalars done',time,dnekclock()-ts
      
      return
      end
c-----------------------------------------------------------------------
      subroutine meshv (igeom)
      
C     Driver for mesh velocity used in conjunction with moving geometry.
C     
C-----------------------------------------------------------------------

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 819 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 819 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 820 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 820 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 821 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 821 "/home/cmaloney111/Nek5000/core/drive2.f"
C     
      IF (IGEOM.EQ.1) RETURN
C     
      IFIELD = 0
      NEL    = NELFLD(IFIELD)
      IMESH  = 1
      IF (IFTMSH(IFIELD)) IMESH = 2
C     
      CALL UPDMSYS (0)
      CALL MVBDRY  (NEL)
      CALL ELASOLV (NEL)
C     
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine time00
c     

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 839 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 839 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 840 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 840 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/CTIMER" 1
c     
c     Timer variables
c     
# 4
      real*8          tmxmf,tmxms,tdsum,taxhm,tcopy,tinvc,tinv3
     $               ,tinit,tadc3,tcol3,ta2s2,tcol2,tadd2
      common /ctimer/ tmxmf,tmxms,tdsum,taxhm,tcopy,tinvc,tinv3
     $               ,tinit,tadc3,tcol3,ta2s2,tcol2,tadd2
c     
      real*8          tsolv,tgsum,tdsnd,tdadd,tcdtp,tmltd,tprep
     $               ,tpres,thmhz,tgop ,tgop1,tdott,tbsol,tbso2
     $               ,tsett,tslvb,tusbc,tddsl,tcrsl,tdsmx,tdsmn
     $               ,tgsmn,tgsmx,teslv,tbbbb,tcccc,tdddd,teeee
     $               ,tvdss,tschw,tadvc,tspro,tgop_sync,tsyc
     $               ,twal,tgp2,tcvf,tproj,tusfq,tuchk
     $               ,tmakf,tmakq
      common /ctime2/ tsolv,tgsum,tdsnd,tdadd,tcdtp,tmltd,tprep
     $               ,tpres,thmhz,tgop ,tgop1,tdott,tbsol,tbso2
     $               ,tsett,tslvb,tusbc,tddsl,tcrsl,tdsmx,tdsmn
     $               ,tgsmn,tgsmx,teslv,tbbbb,tcccc,tdddd,teeee
     $               ,tvdss,tschw,tadvc,tspro,tgop_sync,tsyc
     $               ,twal,tgp2,tcvf,tproj,tusfq,tuchk
     $               ,tmakf,tmakq
c     
      integer nmxmf,nmxms,ndsum,naxhm,ncopy,ninvc,ninv3
      common /itimer/ nmxmf,nmxms,ndsum,naxhm,ncopy,ninvc,ninv3
c     
      integer         nsolv,ngsum,ndsnd,ndadd,ncdtp,nmltd,nprep
     $               ,npres,nhmhz,ngop ,ngop1,ndott,nbsol,nbso2
     $               ,nsett,nslvb,nusbc,nddsl,ncrsl,ndsmx,ndsmn
     $               ,ngsmn,ngsmx,neslv,nbbbb,ncccc,ndddd,neeee
     $               ,nvdss,nadvc,nspro,ngop_sync,nsyc,nwal,ngp2
     $               ,ncvf
      common /itime2/ nsolv,ngsum,ndsnd,ndadd,ncdtp,nmltd,nprep
     $               ,npres,nhmhz,ngop ,ngop1,ndott,nbsol,nbso2
     $               ,nsett,nslvb,nusbc,nddsl,ncrsl,ndsmx,ndsmn
     $               ,ngsmn,ngsmx,neslv,nbbbb,ncccc,ndddd,neeee
     $               ,nvdss,nadvc,nspro,ngop_sync,nsyc,nwal,ngp2
     $               ,ncvf
c     
      real*8          pmxmf,pmxms,pdsum,paxhm,pcopy,pinvc,pinv3
     $               ,psolv,pgsum,pdsnd,pdadd,pcdtp,pmltd,pprep
     $               ,ppres,phmhz,pgop ,pgop1,pdott,pbsol,pbso2
     $               ,psett,pslvb,pusbc,pddsl,pcrsl,pdsmx,pdsmn
     $               ,pgsmn,pgsmx,peslv,pbbbb,pcccc,pdddd,peeee
     $               ,pvdss,pspro,pgop_sync,psyc,pwal,pgp2
      common /ptimer/ pmxmf,pmxms,pdsum,paxhm,pcopy,pinvc,pinv3
     $               ,psolv,pgsum,pdsnd,pdadd,pcdtp,pmltd,pprep
     $               ,ppres,phmhz,pgop ,pgop1,pdott,pbsol,pbso2
     $               ,psett,pslvb,pusbc,pddsl,pcrsl,pdsmx,pdsmn
     $               ,pgsmn,pgsmx,peslv,pbbbb,pcccc,pdddd,peeee
     $               ,pvdss,pspro,pgop_sync,psyc,pwal,pgp2
      
c     
      real*8 etime1,etime2,etime0,gtime1,tscrtch
      real*8 dnekclock,dnekclock_sync
c     
      real*8          etimes,ttotal,tttstp,etims0,ttime
      common /ctime3/ etimes,ttotal,tttstp,etims0,ttime
c     
      integer icalld
      save    icalld
      data    icalld /0/
      
      logical         ifsync
      common /ctimel/ ifsync
# 841 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 841 "/home/cmaloney111/Nek5000/core/drive2.f"
C     
      nmxmf=0
      nmxms=0
      ndsum=0
      nvdss=0
      nsett=0
      ncdtp=0
      npres=0
      nmltd=0
      ngsum=0
      nprep=0
      ndsnd=0
      ndadd=0
      nhmhz=0
      naxhm=0
      ngop =0
      nusbc=0
      ncopy=0
      ninvc=0
      ninv3=0
      nsolv=0
      nslvb=0
      nddsl=0
      ncrsl=0
      ndott=0
      nbsol=0
      nadvc=0
      nspro=0
      ncvf =0
c     
      tmxmf=0.0
      tmxms=0.0
      tdsum=0.0
      tvdss=0.0
      tvdss=0.0
      tdsmn=9.9e9
      tdsmx=0.0
      tsett=0.0
      tcdtp=0.0
      tpres=0.0
      teslv=0.0
      tmltd=0.0
      tgsum=0.0
      tgsmn=9.9e9
      tgsmx=0.0
      tprep=0.0
      tdsnd=0.0
      tdadd=0.0
      thmhz=0.0
      taxhm=0.0
      tgop =0.0
      tusbc=0.0
      tcopy=0.0
      tinvc=0.0
      tinv3=0.0
      tsolv=0.0
      tslvb=0.0
      tddsl=0.0
      tcrsl=0.0
      tdott=0.0
      tbsol=0.0
      tbso2=0.0
      tspro=0.0
      tadvc=0.0
      ttime=0.0
      tcvf =0.0
      tproj=0.0
      tuchk=0.0
      tmakf=0.0
      tmakq=0.0
C     
      return
      end
C     
c-----------------------------------------------------------------------
      subroutine runstat
      

      

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 921 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 921 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 922 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 922 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/CTIMER" 1
c     
c     Timer variables
c     
# 4
      real*8          tmxmf,tmxms,tdsum,taxhm,tcopy,tinvc,tinv3
     $               ,tinit,tadc3,tcol3,ta2s2,tcol2,tadd2
      common /ctimer/ tmxmf,tmxms,tdsum,taxhm,tcopy,tinvc,tinv3
     $               ,tinit,tadc3,tcol3,ta2s2,tcol2,tadd2
c     
      real*8          tsolv,tgsum,tdsnd,tdadd,tcdtp,tmltd,tprep
     $               ,tpres,thmhz,tgop ,tgop1,tdott,tbsol,tbso2
     $               ,tsett,tslvb,tusbc,tddsl,tcrsl,tdsmx,tdsmn
     $               ,tgsmn,tgsmx,teslv,tbbbb,tcccc,tdddd,teeee
     $               ,tvdss,tschw,tadvc,tspro,tgop_sync,tsyc
     $               ,twal,tgp2,tcvf,tproj,tusfq,tuchk
     $               ,tmakf,tmakq
      common /ctime2/ tsolv,tgsum,tdsnd,tdadd,tcdtp,tmltd,tprep
     $               ,tpres,thmhz,tgop ,tgop1,tdott,tbsol,tbso2
     $               ,tsett,tslvb,tusbc,tddsl,tcrsl,tdsmx,tdsmn
     $               ,tgsmn,tgsmx,teslv,tbbbb,tcccc,tdddd,teeee
     $               ,tvdss,tschw,tadvc,tspro,tgop_sync,tsyc
     $               ,twal,tgp2,tcvf,tproj,tusfq,tuchk
     $               ,tmakf,tmakq
c     
      integer nmxmf,nmxms,ndsum,naxhm,ncopy,ninvc,ninv3
      common /itimer/ nmxmf,nmxms,ndsum,naxhm,ncopy,ninvc,ninv3
c     
      integer         nsolv,ngsum,ndsnd,ndadd,ncdtp,nmltd,nprep
     $               ,npres,nhmhz,ngop ,ngop1,ndott,nbsol,nbso2
     $               ,nsett,nslvb,nusbc,nddsl,ncrsl,ndsmx,ndsmn
     $               ,ngsmn,ngsmx,neslv,nbbbb,ncccc,ndddd,neeee
     $               ,nvdss,nadvc,nspro,ngop_sync,nsyc,nwal,ngp2
     $               ,ncvf
      common /itime2/ nsolv,ngsum,ndsnd,ndadd,ncdtp,nmltd,nprep
     $               ,npres,nhmhz,ngop ,ngop1,ndott,nbsol,nbso2
     $               ,nsett,nslvb,nusbc,nddsl,ncrsl,ndsmx,ndsmn
     $               ,ngsmn,ngsmx,neslv,nbbbb,ncccc,ndddd,neeee
     $               ,nvdss,nadvc,nspro,ngop_sync,nsyc,nwal,ngp2
     $               ,ncvf
c     
      real*8          pmxmf,pmxms,pdsum,paxhm,pcopy,pinvc,pinv3
     $               ,psolv,pgsum,pdsnd,pdadd,pcdtp,pmltd,pprep
     $               ,ppres,phmhz,pgop ,pgop1,pdott,pbsol,pbso2
     $               ,psett,pslvb,pusbc,pddsl,pcrsl,pdsmx,pdsmn
     $               ,pgsmn,pgsmx,peslv,pbbbb,pcccc,pdddd,peeee
     $               ,pvdss,pspro,pgop_sync,psyc,pwal,pgp2
      common /ptimer/ pmxmf,pmxms,pdsum,paxhm,pcopy,pinvc,pinv3
     $               ,psolv,pgsum,pdsnd,pdadd,pcdtp,pmltd,pprep
     $               ,ppres,phmhz,pgop ,pgop1,pdott,pbsol,pbso2
     $               ,psett,pslvb,pusbc,pddsl,pcrsl,pdsmx,pdsmn
     $               ,pgsmn,pgsmx,peslv,pbbbb,pcccc,pdddd,peeee
     $               ,pvdss,pspro,pgop_sync,psyc,pwal,pgp2
      
c     
      real*8 etime1,etime2,etime0,gtime1,tscrtch
      real*8 dnekclock,dnekclock_sync
c     
      real*8          etimes,ttotal,tttstp,etims0,ttime
      common /ctime3/ etimes,ttotal,tttstp,etims0,ttime
c     
      integer icalld
      save    icalld
      data    icalld /0/
      
      logical         ifsync
      common /ctimel/ ifsync
# 923 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 923 "/home/cmaloney111/Nek5000/core/drive2.f"
      
      real min_dsum, max_dsum, avg_dsum
      real min_vdss, max_vdss, avg_vdss
      real min_gop,  max_gop,  avg_gop
      real min_gop_sync,  max_gop_sync,  avg_gop_sync
      real min_crsl, max_crsl, avg_crsl
      real min_usbc, max_usbc, avg_usbc
      real min_syc, max_syc, avg_syc
      real min_wal, max_wal, avg_wal
      real min_irc, max_irc, avg_irc
      real min_isd, max_isd, avg_isd
      real min_comm, max_comm, avg_comm
      
      real comm_timers(8)
      integer comm_counters(8)
      character*132 s132
      
      tstop=dnekclock()
      tttstp=ttime         ! sum over all timesteps
      
c      call opcount(3)      ! print op-counters
      
      tcomm  = tisd + tirc + tsyc + tgp2+ twal + trc + tsd
      min_comm = tcomm
      call gop(min_comm,wwork,'m  ',1)
      max_comm = tcomm
      call gop(max_comm,wwork,'M  ',1)
      avg_comm = tcomm
      call gop(avg_comm,wwork,'+  ',1)
      avg_comm = avg_comm/np
c     
      min_isd = tisd
      call gop(min_isd,wwork,'m  ',1)
      max_isd = tisd
      call gop(max_isd,wwork,'M  ',1)
      avg_isd = tisd
      call gop(avg_isd,wwork,'+  ',1)
      avg_isd = avg_isd/np
c     
      min_irc = tirc
      call gop(min_irc,wwork,'m  ',1)
      max_irc = tirc
      call gop(max_irc,wwork,'M  ',1)
      avg_irc = tirc
      call gop(avg_irc,wwork,'+  ',1)
      avg_irc = avg_irc/np
c     
      min_syc = tsyc
      call gop(min_syc,wwork,'m  ',1)
      max_syc = tsyc
      call gop(max_syc,wwork,'M  ',1)
      avg_syc = tsyc
      call gop(avg_syc,wwork,'+  ',1)
      avg_syc = avg_syc/np
c     
      min_wal = twal
      call gop(min_wal,wwork,'m  ',1)
      max_wal = twal
      call gop(max_wal,wwork,'M  ',1)
      avg_wal = twal
      call gop(avg_wal,wwork,'+  ',1)
      avg_wal = avg_wal/np
c     
      min_gop = tgp2
      call gop(min_gop,wwork,'m  ',1)
      max_gop = tgp2
      call gop(max_gop,wwork,'M  ',1)
      avg_gop = tgp2
      call gop(avg_gop,wwork,'+  ',1)
      avg_gop = avg_gop/np
c     
      min_gop_sync = tgop_sync
      call gop(min_gop_sync,wwork,'m  ',1)
      max_gop_sync = tgop_sync
      call gop(max_gop_sync,wwork,'M  ',1)
      avg_gop_sync = tgop_sync
      call gop(avg_gop_sync,wwork,'+  ',1)
      avg_gop_sync = avg_gop_sync/np
c     
      min_vdss = tvdss
      call gop(min_vdss,wwork,'m  ',1)
      max_vdss = tvdss
      call gop(max_vdss,wwork,'M  ',1)
      avg_vdss = tvdss
      call gop(avg_vdss,wwork,'+  ',1)
      avg_vdss = avg_vdss/np
c     
      min_dsum = tdsum
      call gop(min_dsum,wwork,'m  ',1)
      max_dsum = tdsum
      call gop(max_dsum,wwork,'M  ',1)
      avg_dsum = tdsum
      call gop(avg_dsum,wwork,'+  ',1)
      avg_dsum = avg_dsum/np
c     
      
      min_crsl = tcrsl
      call gop(min_crsl,wwork,'m  ',1)
      max_crsl = tcrsl
      call gop(max_crsl,wwork,'M  ',1)
      avg_crsl = tcrsl
      call gop(avg_crsl,wwork,'+  ',1)
      avg_crsl = avg_crsl/np
c     
      min_usbc = tusbc
      call gop(min_usbc,wwork,'m  ',1)
      max_usbc = tusbc
      call gop(max_usbc,wwork,'M  ',1)
      avg_usbc = tusbc
      call gop(avg_usbc,wwork,'+  ',1)
      avg_usbc = avg_usbc/np
c     
      tttstp = tttstp + 1e-7
      if (nio.eq.0) then
         write(6,*) ''
         write(6,'(A)') 'runtime statistics:'
      
         pinit=tinit/tttstp
         write(6,*) 'init time',tinit,pinit
         pprep=tprep/tttstp
         write(6,*) 'prep time',nprep,tprep,pprep
      
c        Pressure solver timings
         ppres=tpres/tttstp
         write(6,*) 'pres time',npres,tpres,ppres
      
c        Coarse grid solver timings
         pcrsl=tcrsl/tttstp
         write(6,*) 'crsl time',ncrsl,tcrsl,pcrsl
         write(6,*) 'crsl min ',min_crsl
         write(6,*) 'crsl max ',max_crsl
         write(6,*) 'crsl avg ',avg_crsl
      
c        Helmholz solver timings
         phmhz=thmhz/tttstp
         write(6,*) 'hmhz time',nhmhz,thmhz,phmhz
      
c        E solver timings
         peslv=teslv/tttstp 
         write(6,*) 'eslv time',neslv,teslv,peslv
      
c        makef timings
         pmakf=tmakf/tttstp 
         write(6,*) 'makf time',tmakf,pmakf
      
c        makeq timings
         pmakq=tmakq/tttstp 
         write(6,*) 'makq time',tmakq,pmakq
      
c        CVODE RHS timings
         pcvf=tcvf/tttstp
         if(ifcvode) write(6,*) 'cfun time',ncvf,tcvf,pcvf
      
c        Resiual projection timings
         pproj=tproj/tttstp
         write(6,*) 'proj time',tproj,pproj
      
c        Variable properties timings
         pspro=tspro/tttstp
         write(6,*) 'usvp time',nspro,tspro,pspro
      
c        User q and f timings
         pusfq=tusfq/tttstp
         write(6,*) 'usfq time',0,tusfq,pusfq
      
c        USERBC timings
         pusbc=tusbc/tttstp
         write(6,*) 'usbc time',nusbc,tusbc,pusbc
         write(6,*) 'usbc min ',min_usbc 
         write(6,*) 'usbc max ',max_usbc 
         write(6,*) 'usb  avg ',avg_usbc 
      
c        User check timings
         puchk=tuchk/tttstp
         write(6,*) 'uchk time',tuchk,puchk
      
c        Operator timings
         pmltd=tmltd/tttstp
         write(6,*) 'mltd time',nmltd,tmltd,pmltd
         pcdtp=tcdtp/tttstp
         write(6,*) 'cdtp time',ncdtp,tcdtp,pcdtp
         paxhm=taxhm/tttstp
         write(6,*) 'axhm time',naxhm,taxhm,paxhm
         padvc=tadvc/tttstp
         write(6,*) 'advc time',nadvc,tadvc,padvc
      
c        Low-level routines
         pmxmf=tmxmf/tttstp
         write(6,*) 'mxmf time',tmxmf,pmxmf
         padc3=tadc3/tttstp
         write(6,*) 'adc3 time',tadc3,padc3
         pcol2=tcol2/tttstp
         write(6,*) 'col2 time',tcol2,pcol2
         pcol3=tcol3/tttstp
         write(6,*) 'col3 time',tcol3,pcol3
         pa2s2=ta2s2/tttstp
         write(6,*) 'a2s2 time',ta2s2,pa2s2
         padd2=tadd2/tttstp
         write(6,*) 'add2 time',tadd2,padd2
         pinvc=tinvc/tttstp
         write(6,*) 'invc time',tinvc,pinvc
      
c         pinv3=tinv3/tttstp
c         write(6,*) 'inv3 time',ninv3,tinv3,pinv3
      
         pgop=tgop/tttstp
         write(6,*) 'tgop time',ngop,tgop,pgop
      
         pdadd=tdadd/tttstp
         write(6,*) 'dadd time',ndadd,tdadd,pdadd
      
c        Vector direct stiffness summuation timings
         pvdss=tvdss/tttstp
         write(6,*) 'vdss time',nvdss,tvdss,pvdss
         write(6,*) 'vdss min ',min_vdss
         write(6,*) 'vdss max ',max_vdss
         write(6,*) 'vdss avg ',avg_vdss
      
c        Direct stiffness summuation timings
         pdsum=tdsum/tttstp
         write(6,*) 'dsum time',ndsum,tdsum,pdsum
         write(6,*) 'dsum min ',min_dsum
         write(6,*) 'dsum max ',max_dsum
         write(6,*) 'dsum avg ',avg_dsum
      
c         pgsum=tgsum/tttstp
c         write(6,*) 'gsum time',ngsum,tgsum,pgsum
      
c         pdsnd=tdsnd/tttstp
c         write(6,*) 'dsnd time',ndsnd,tdsnd,pdsnd
      
c         pdsmx=tdsmx/tttstp
c         write(6,*) 'dsmx time',ndsmx,tdsmx,pdsmx
c         pdsmn=tdsmn/tttstp
c         write(6,*) 'dsmn time',ndsmn,tdsmn,pdsmn
c         pslvb=tslvb/tttstp
c         write(6,*) 'slvb time',nslvb,tslvb,pslvb
      
         pddsl=tddsl/tttstp
         write(6,*) 'ddsl time',nddsl,tddsl,pddsl
      
c         pbsol=tbsol/tttstp
c         write(6,*) 'bsol time',nbsol,tbsol,pbsol
c         pbso2=tbso2/tttstp
c         write(6,*) 'bso2 time',nbso2,tbso2,pbso2
      
         write(6,*) ''
      endif
      
      if (lastep.eq.1) then
        if (nio.eq.0)  ! header for timing
     $    write(6,1) 'tusbc','tdadd','tcrsl','tvdss','tdsum',
     $               ' tgop',ifsync
    1     format(/,'#',2x,'nid',6(7x,a5),4x,'qqq',1x,l4)
      
        call blank(s132,132)
        write(s132,132) nid,tusbc,tdadd,tcrsl,tvdss,tdsum,tgop
  132   format(i12,1p6e12.4,' qqq')
        call pprint_all(s132,132,6)
      endif

      
# 1185
      return
      end
c-----------------------------------------------------------------------
      subroutine pprint_all(s,n_in,io)
      character*1 s(n_in)
      character*1 w(132)
      

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 1193 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1193 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 1194 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1194 "/home/cmaloney111/Nek5000/core/drive2.f"
      
      n = min(132,n_in)
      
      mtag = 999
      m    = 1
      call nekgsync()
      
      if (nid.eq.0) then
         l = ltrunc(s,n)
         write(io,1) (s(k),k=1,l)
   1     format(132a1)
      
         do i=1,np-1
            call csend(mtag,s,1,i,0)    ! send handshake
            m = 132
            call blank(w,m)
            call crecv(i,w,m)
            if (m.le.132) then
               l = ltrunc(w,m)
               write(io,1) (w(k),k=1,l)
            else
               write(io,*) 'pprint long message: ',i,m
               l = ltrunc(w,132)
               write(io,1) (w(k),k=1,l)
            endif
         enddo
      else
         call crecv(mtag,w,m)          ! wait for handshake
         l = ltrunc(s,n)
         call csend(nid,s,l,0,0)       ! send data to node 0
      endif
      return
      end
c-----------------------------------------------------------------------
      
      subroutine opcount(ICALL)
C     

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 1232 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1232 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 1233 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1233 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/OPCTR" 1
c     
C     OPCTR is a set of arrays for tracking the number of operations,
C     and number of calls for a particular subroutine
      
# 5
      integer maxrts
      parameter (maxrts=1000)
      
      character*6     rname(maxrts)
      common /opctrc/ rname
c     
      real*8          dct(maxrts),rct(maxrts),dcount
      common /opctrd/ dct        ,rct        ,dcount
c     
      integer         ncall(maxrts),nrout
      common /opctri/ ncall        ,nrout
c     
      integer myrout,isclld
      save    myrout,isclld
      data    myrout /0/
      data    isclld /0/
# 1234 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1234 "/home/cmaloney111/Nek5000/core/drive2.f"
      
      character*6 sname(maxrts)
      integer     ind  (maxrts)
      integer     idum (maxrts)
C     
      if (icall.eq.1) then
         nrout=0
      endif
      if (icall.eq.1.or.icall.eq.2) then
         dcount = 0.0
         do 100 i=1,maxrts
            ncall(i) = 0
            dct(i)   = 0.0
  100    continue
      endif
      if (icall.eq.3) then
C     
C        Sort and print out diagnostics
C     
         if (nid.eq.0) then
            write(6,*) nid,' opcount',dcount
            do i = 1,np-1
              call csend(i,idum,4,i,0) 
              call crecv(i,ddcount,wdsize)
               write(6,*) i,' opcount',ddcount
            enddo
         else
            call crecv (nid,idum,4)
            call csend (nid,dcount,wdsize,0,0) 
         endif
      
         dhc = dcount
         call gop(dhc,dwork,'+  ',1)
         if (nio.eq.0) then
            write(6,*) ' TOTAL OPCOUNT',dhc
         endif
C     
         CALL DRCOPY(rct,dct,nrout)
         CALL SORT(rct,ind,nrout)
         CALL CHSWAPR(rname,6,ind,nrout,sname)
         call iswap(ncall,ind,nrout,idum)
C     
         if (nio.eq.0) then
            do 200 i=1,nrout
               write(6,201) nid,rname(i),rct(i),ncall(i)
  200       continue
  201       format(2x,' opnode',i4,2x,a6,g18.7,i12)
         endif
      endif
      return
      end
C     
c-----------------------------------------------------------------------
      subroutine dofcnt

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 1289 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1289 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 1290 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1290 "/home/cmaloney111/Nek5000/core/drive2.f"
      COMMON /SCRNS/ WORK(LCTMP1)
      
      integer*8 ntot,ntotp,ntotv
      
      nxyz  = nx1*ny1*nz1
      nel   = nelv
      
      ! unique points on v-mesh
      vpts = glsum(vmult,nel*nxyz) + .1
      nvtot=vpts
      
      ! unique points on pressure mesh
      work(1)=nel*nxyz
      ppts = glsum(work,1) + .1
      ntot=ppts
C     
      if (nio.eq.0) write(6,'(A,2i13)')
     &   'gridpoints unique/tot: ',nvtot,ntot
      
      ntot1=nx1*ny1*nz1*nelv
      ntot2=nx2*ny2*nz2*nelv
      
      ntotv = glsc2(tmult,tmask,ntot1)
      ntotp = i8glsum(ntot2,1)
      
      if (ifflow)  ntotv = glsc2(vmult,v1mask,ntot1) + .1
      if (ifsplit) ntotp = glsc2(vmult,pmask ,ntot1) + .1
      if (nio.eq.0) write(6,'(A,2i13)') 
     $   'dofs vel/pr:           ',ntotv,ntotp
      
      return
      end
c-----------------------------------------------------------------------
      subroutine vol_flow
c     
c     
c     Adust flow volume at end of time step to keep flow rate fixed by
c     adding an appropriate multiple of the linear solution to the Stoke
c     problem arising from a unit forcing in the X-direction.  This assu
c     that the flow rate in the X-direction is to be fixed (as opposed t
c     or Z-) *and* that the periodic boundary conditions in the X-direct
c     occur at the extreme left and right ends of the mesh.
c     
c     pff 6/28/98
c     

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 1336 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1336 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 1337 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1337 "/home/cmaloney111/Nek5000/core/drive2.f"
c     
c     Swap the comments on these two lines if you don't want to fix the
c     flow rate for periodic-in-X (or Z) flow problems.
c     
      parameter (kx1=lx1,ky1=ly1,kz1=lz1,kx2=lx2,ky2=ly2,kz2=lz2)
c     
      common /cvflow_a/ vxc(kx1,ky1,kz1,lelv)
     $                , vyc(kx1,ky1,kz1,lelv)
     $                , vzc(kx1,ky1,kz1,lelv)
     $                , prc(kx2,ky2,kz2,lelv)
     $                , vdc(kx1*ky1*kz1*lelv,2)
      common /cvflow_r/ flow_rate,base_flow,domain_length,xsec
     $                , scale_vf(3)
      common /cvflow_i/ icvflow,iavflow
      common /cvflow_c/ chv(3)
      character*1 chv
c     
      real bd_vflow,dt_vflow
      save bd_vflow,dt_vflow
      data bd_vflow,dt_vflow /-99.,-99./
      
      logical ifcomp
      
c     Check list:
      
c     param (55) -- volume flow rate, if nonzero
c     forcing in X? or in Z?
      
      
      ntot1 = lx1*ly1*lz1*nelv
      ntot2 = lx2*ly2*lz2*nelv
      
      if (param(55).eq.0.) return
      if (kx1.eq.1) then
         write(6,*) 'ABORT. Recompile vol_flow with kx1=lx1, etc.'
         call exitt
      endif
      
      icvflow   = 1                                  ! Default flow dir.
      if (param(54).ne.0) icvflow = abs(param(54))
      iavflow   = 0                                  ! Determine flow ra
      if (param(54).lt.0) iavflow = 1                ! from mean velocit
      flow_rate = param(55)
      
      chv(1) = 'X'
      chv(2) = 'Y'
      chv(3) = 'Z'
      
c     If either dt or the backwards difference coefficient change,
c     then recompute base flow solution corresponding to unit forcing:
      
      ifcomp = .false.
      if (dt.ne.dt_vflow.or.bd(1).ne.bd_vflow.or.ifmvbd) ifcomp=.true.
      if (.not.ifcomp) then
         ifcomp=.true.
         do i=1,ntot1
            if (vdiff (i,1,1,1,1).ne.vdc(i,1)) goto 20
            if (vtrans(i,1,1,1,1).ne.vdc(i,2)) goto 20
         enddo
         ifcomp=.false.  ! If here, then vdiff/vtrans unchanged.
   20    continue
      endif
      call gllog(ifcomp,.true.)
      
      call copy(vdc(1,1),vdiff (1,1,1,1,1),ntot1)
      call copy(vdc(1,2),vtrans(1,1,1,1,1),ntot1)
      dt_vflow = dt
      bd_vflow = bd(1)
      
      if (ifcomp) call compute_vol_soln(vxc,vyc,vzc,prc)
      
      if (icvflow.eq.1) current_flow=glsc2(vx,bm1,ntot1)/domain_length  
      if (icvflow.eq.2) current_flow=glsc2(vy,bm1,ntot1)/domain_length  
      if (icvflow.eq.3) current_flow=glsc2(vz,bm1,ntot1)/domain_length  
      
      if (iavflow.eq.1) then
         xsec = volvm1 / domain_length
         flow_rate = param(55)*xsec
      endif
      
      delta_flow = flow_rate-current_flow
      
c     Note, this scale factor corresponds to FFX, provided FFX has
c     not also been specified in userf.   If ffx is also specified
c     in userf then the true FFX is given by ffx_userf + scale.
      
      scale = delta_flow/base_flow
      scale_vf(icvflow) = scale
      if (nio.eq.0) write(6,1) istep,chv(icvflow)
     $   ,time,scale,delta_flow,current_flow,flow_rate
    1    format(i11,'  Volflow ',a1,11x,1p5e13.4)
      
      call add2s2(vx,vxc,scale,ntot1)
      call add2s2(vy,vyc,scale,ntot1)
      call add2s2(vz,vzc,scale,ntot1)
      call add2s2(pr,prc,scale,ntot2)
      
      return
      end
c-----------------------------------------------------------------------
      subroutine compute_vol_soln(vxc,vyc,vzc,prc)
c     
c     Compute the solution to the time-dependent Stokes problem
c     with unit forcing, and find associated flow rate.
c     
c     pff 2/28/98
c     

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 1445 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1445 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 1446 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1446 "/home/cmaloney111/Nek5000/core/drive2.f"
c     
      real vxc(lx1,ly1,lz1,lelv)
     $   , vyc(lx1,ly1,lz1,lelv)
     $   , vzc(lx1,ly1,lz1,lelv)
     $   , prc(lx2,ly2,lz2,lelv)
c     
      common /cvflow_r/ flow_rate,base_flow,domain_length,xsec
     $                , scale_vf(3)
      common /cvflow_i/ icvflow,iavflow
      common /cvflow_c/ chv(3)
      character*1 chv
c     
      integer icalld
      save    icalld
      data    icalld/0/
c     
c     
      ntot1 = lx1*ly1*lz1*nelv
      if (icalld.eq.0) then
         icalld=icalld+1
         xlmin = glmin(xm1,ntot1)
         xlmax = glmax(xm1,ntot1)
         ylmin = glmin(ym1,ntot1)          !  for Y!
         ylmax = glmax(ym1,ntot1)
         zlmin = glmin(zm1,ntot1)          !  for Z!
         zlmax = glmax(zm1,ntot1)
c     
         if (icvflow.eq.1) domain_length = xlmax - xlmin
         if (icvflow.eq.2) domain_length = ylmax - ylmin
         if (icvflow.eq.3) domain_length = zlmax - zlmin
c     
      endif
c     
      if (ifsplit) then
c        call plan2_vol(vxc,vyc,vzc,prc)
         call plan4_vol(vxc,vyc,vzc,prc)
      else
         call plan3_vol(vxc,vyc,vzc,prc)
      endif
c     
c     Compute base flow rate
c     
      if (icvflow.eq.1) base_flow = glsc2(vxc,bm1,ntot1)/domain_length
      if (icvflow.eq.2) base_flow = glsc2(vyc,bm1,ntot1)/domain_length
      if (icvflow.eq.3) base_flow = glsc2(vzc,bm1,ntot1)/domain_length
c     
      if (nio.eq.0 .and. loglevel.gt.2) write(6,1) 
     $   istep,chv(icvflow),base_flow,domain_length,flow_rate
    1    format(i11,'  basflow ',a1,11x,1p3e13.4)
c     
      return
      end
c-----------------------------------------------------------------------
      subroutine plan2_vol(vxc,vyc,vzc,prc)
c     
c     Compute pressure and velocity using fractional step method.
c     (classical splitting scheme).
c     
c     

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 1506 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1506 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 1507 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1507 "/home/cmaloney111/Nek5000/core/drive2.f"
c     
      real vxc(lx1,ly1,lz1,lelv)
     $   , vyc(lx1,ly1,lz1,lelv)
     $   , vzc(lx1,ly1,lz1,lelv)
     $   , prc(lx2,ly2,lz2,lelv)
C     
      COMMON /SCRNS/ RESV1 (LX1,LY1,LZ1,LELV)
     $ ,             RESV2 (LX1,LY1,LZ1,LELV)
     $ ,             RESV3 (LX1,LY1,LZ1,LELV)
     $ ,             RESPR (LX2,LY2,LZ2,LELV)
      COMMON /SCRVH/ H1    (LX1,LY1,LZ1,LELV)
     $ ,             H2    (LX1,LY1,LZ1,LELV)
c     
      common /cvflow_i/ icvflow,iavflow
C     
C     
C     Compute pressure 
C     
      ntot1  = lx1*ly1*lz1*nelv
c     
      if (icvflow.eq.1) then
         call cdtp     (respr,v1mask,rxm2,sxm2,txm2,1)
      elseif (icvflow.eq.2) then
         call cdtp     (respr,v2mask,rxm2,sxm2,txm2,1)
      else
         call cdtp     (respr,v3mask,rxm2,sxm2,txm2,1)
      endif
c     
      call ortho    (respr)
c     
      call ctolspl  (tolspl,respr)
      call rone     (h1,ntot1)
      call rzero    (h2,ntot1)
c     
      call hmholtz  ('PRES',prc,respr,h1,h2,pmask,vmult,
     $                             imesh,tolspl,nmxp,1)
      call ortho    (prc)
C     
C     Compute velocity
C     
      call opgrad   (resv1,resv2,resv3,prc)
      call opchsgn  (resv1,resv2,resv3)
      call add2col2 (resv1,bm1,v1mask,ntot1)
c     
      intype = -1
      call sethlm   (h1,h2,intype)
      call ophinv   (vxc,vyc,vzc,resv1,resv2,resv3,h1,h2,tolhv,nmxv)
C     
      return
      end
c-----------------------------------------------------------------------
      subroutine plan3_vol(vxc,vyc,vzc,prc)
c     
c     Compute pressure and velocity using fractional step method.
c     (PLAN3).
c     
c     

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 1565 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1565 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 1566 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1566 "/home/cmaloney111/Nek5000/core/drive2.f"
c     
      real vxc(lx1,ly1,lz1,lelv)
     $   , vyc(lx1,ly1,lz1,lelv)
     $   , vzc(lx1,ly1,lz1,lelv)
     $   , prc(lx2,ly2,lz2,lelv)
C     
      COMMON /SCRNS/ rw1   (LX1,LY1,LZ1,LELV)
     $ ,             rw2   (LX1,LY1,LZ1,LELV)
     $ ,             rw3   (LX1,LY1,LZ1,LELV)
     $ ,             dv1   (LX1,LY1,LZ1,LELV)
     $ ,             dv2   (LX1,LY1,LZ1,LELV)
     $ ,             dv3   (LX1,LY1,LZ1,LELV)
     $ ,             RESPR (LX2,LY2,LZ2,LELV)
      COMMON /SCRVH/ H1    (LX1,LY1,LZ1,LELV)
     $ ,             H2    (LX1,LY1,LZ1,LELV)
      COMMON /SCRHI/ H2INV (LX1,LY1,LZ1,LELV)
      common /cvflow_i/ icvflow,iavflow
c     
c     
c     Compute velocity, 1st part 
c     
      ntot1  = lx1*ly1*lz1*nelv
      ntot2  = lx2*ly2*lz2*nelv
      ifield = 1
c     
      if (icvflow.eq.1) then
         call copy     (rw1,bm1,ntot1)
         call rzero    (rw2,ntot1)
         call rzero    (rw3,ntot1)
      elseif (icvflow.eq.2) then
         call rzero    (rw1,ntot1)
         call copy     (rw2,bm1,ntot1)
         call rzero    (rw3,ntot1)
      else
         call rzero    (rw1,ntot1)        ! Z-flow!
         call rzero    (rw2,ntot1)        ! Z-flow!
         call copy     (rw3,bm1,ntot1)    ! Z-flow!
      endif
      intype = -1
      call sethlm   (h1,h2,intype)
      call ophinv   (vxc,vyc,vzc,rw1,rw2,rw3,h1,h2,tolhv,nmxv)
      call ssnormd  (vxc,vyc,vzc)
c     
c     Compute pressure  (from "incompr")
c     
      intype = 1
      dtinv  = 1./dt
c     
      call rzero   (h1,ntot1)
      call copy    (h2,vtrans(1,1,1,1,ifield),ntot1)
      call cmult   (h2,dtinv,ntot1)
      call invers2 (h2inv,h2,ntot1)
      call opdiv   (respr,vxc,vyc,vzc)
      call chsign  (respr,ntot2)
      call ortho   (respr)
c     
c     
c     Set istep=0 so that h1/h2 will be re-initialized in eprec
      i_tmp = istep
      istep = 0
      call esolver (respr,h1,h2,h2inv,intype)
      istep = i_tmp
c     
      call opgradt (rw1,rw2,rw3,respr)
      call opbinv  (dv1,dv2,dv3,rw1,rw2,rw3,h2inv)
      call opadd2  (vxc,vyc,vzc,dv1,dv2,dv3)
c     
      call cmult2  (prc,respr,bd(1),ntot2)
c     
      return
      end
c-----------------------------------------------------------------------
      subroutine plan4_vol(vxc,vyc,vzc,prc)
      
c     Compute pressure and velocity using fractional step method.
c     (Tombo splitting scheme).
      
      
      

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 1646 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1646 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 1647 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1647 "/home/cmaloney111/Nek5000/core/drive2.f"
      
      real vxc(lx1,ly1,lz1,lelv)
     $   , vyc(lx1,ly1,lz1,lelv)
     $   , vzc(lx1,ly1,lz1,lelv)
     $   , prc(lx1,ly1,lz1,lelv)
      
      common /scrns/ resv1 (lx1,ly1,lz1,lelv)
     $ ,             resv2 (lx1,ly1,lz1,lelv)
     $ ,             resv3 (lx1,ly1,lz1,lelv)
     $ ,             respr (lx1,ly1,lz1,lelv)
      common /scrvh/ h1    (lx1,ly1,lz1,lelv)
     $ ,             h2    (lx1,ly1,lz1,lelv)
      
      common /cvflow_i/ icvflow,iavflow
      
      n = lx1*ly1*lz1*nelv
      call invers2  (h1,vtrans,n)
      call rzero    (h2,       n)
      
c     Compute pressure 
      
      if (icvflow.eq.1) call cdtp(respr,h1,rxm2,sxm2,txm2,1)
      if (icvflow.eq.2) call cdtp(respr,h1,rym2,sym2,tym2,1)
      if (icvflow.eq.3) call cdtp(respr,h1,rzm2,szm2,tzm2,1)
      
      call ortho    (respr)
      call ctolspl  (tolspl,respr)
      
      call hmholtz  ('PRES',prc,respr,h1,h2,pmask,vmult,
     $                             imesh,tolspl,nmxp,1)
      call ortho    (prc)
      
C     Compute velocity
      
      call opgrad   (resv1,resv2,resv3,prc)
      if (ifaxis) call col2 (resv2,omask,n)
      call opchsgn  (resv1,resv2,resv3)
      
      if (icvflow.eq.1) call add2col2(resv1,v1mask,bm1,n) ! add forcing
      if (icvflow.eq.2) call add2col2(resv2,v2mask,bm1,n)
      if (icvflow.eq.3) call add2col2(resv3,v3mask,bm1,n)
      
      
      if (ifexplvis) call split_vis ! split viscosity into exp/imp part
      
      intype = -1
      call sethlm   (h1,h2,intype)
      call ophinv   (vxc,vyc,vzc,resv1,resv2,resv3,h1,h2,tolhv,nmxv)
      
      if (ifexplvis) call redo_split_vis ! restore vdiff
      
      end
c-----------------------------------------------------------------------
      subroutine a_dmp
c     

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 1703 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1703 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 1704 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1704 "/home/cmaloney111/Nek5000/core/drive2.f"
      COMMON /SCRNS/ w(LX1,LY1,LZ1,LELT)
      COMMON /SCRUZ/ v (LX1,LY1,LZ1,LELT)
     $             , h1(LX1,LY1,LZ1,LELT)
     $             , h2(LX1,LY1,LZ1,LELT)
c     
      ntot = lx1*ly1*lz1*nelv
      call rone (h1,ntot)
      call rzero(h2,ntot)
      do i=1,ntot
         call rzero(v,ntot)
         v(i,1,1,1) = 1.
         call axhelm (w,v,h1,h2,1,1)
         call outrio (w,ntot,55)
      enddo
c     write(6,*) 'quit in a_dmp'
c     call exitt
      return
      end
c-----------------------------------------------------------------------
      subroutine outrio (v,n,io)
c     
      real v(1)
c     
      write(6,*) 'outrio:',n,io,v(1)
      write(io,6) (v(k),k=1,n)
    6 format(1pe19.11)
c     
c     nr = min(12,n)
c     write(io,6) (v(k),k=1,nr)
c   6 format(1p12e11.3)
      return
      end
c-----------------------------------------------------------------------
      subroutine reset_prop
C-----------------------------------------------------------------------
C     
C     Set variable property arrays
C     
C-----------------------------------------------------------------------

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 1744 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1744 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 1745 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1745 "/home/cmaloney111/Nek5000/core/drive2.f"
C     
C     Caution: 2nd and 3rd strainrate invariants residing in scratch
C              common /SCREV/ are used in STNRINV and NEKASGN
C     
      COMMON /SCREV/ SII (LX1,LY1,LZ1,LELT)
     $             , SIII(LX1,LY1,LZ1,LELT)
      COMMON /SCRUZ/ TA(LX1,LY1,LZ1,LELT)
C     
      real    rstart
      save    rstart
      data    rstart  /1/
c     
      rfinal   = 1./param(2) ! Target Re
c     
      ntot  = lx1*ly1*lz1*nelv
      iramp = 200
      istpp = istep
c     istpp = istep+2033+1250
      if (istpp.ge.iramp) then
         vfinal=1./rfinal
         call cfill(vdiff,vfinal,ntot)
      else
         one = 1.
         pi2 = 2.*atan(one)
         sarg  = (pi2*istpp)/iramp
         sarg  = sin(sarg)
         rnew = rstart + (rfinal-rstart)*sarg
         vnew = 1./rnew
         call cfill(vdiff,vnew,ntot)
         if (nio.eq.0) write(6,*) istep,' New Re:',rnew,sarg,istpp
      endif
      return
      end
C-----------------------------------------------------------------------
      subroutine prinit
      

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/SIZE" 1
c     Include file to dimension static arrays
c     and to set some hardwired run-time parameters
c     
# 4
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lelt,lpmin,ldimt
      integer lpelt,lbelt,toteq,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,lpert,nsessmax,lxo
      integer lfdm,ldimt_proj,lelr
      
      ! BASIC
      parameter (ldim=2)               ! domain dimension (2 or 3)
      parameter (lx1=8)                ! GLL points per element along ea
      parameter (lxd=12)               ! GL  points for over-integration
      parameter (lx2=lx1)            ! GLL points for pressure (lx1 or l
      
      parameter (lelg=20000)            ! max number of global elements
      parameter (lpmin=2)              ! min number of MPI ranks
      parameter (lelt=20000)  ! max number of local elements per MPI ran
      parameter (ldimt=4)              ! max auxiliary fields (temperatu
      
      ! OPTIONAL
      parameter (ldimt_proj=1)         ! max auxiliary fields residual p
      parameter (lelr=lelt)            ! max number of local elements pe
      parameter (lhis=100)               ! max history/monitoring points
      parameter (maxobj=1)             ! max number of objects
      parameter (lpert=1)              ! max number of perturbations
      parameter (toteq=1)              ! max number of conserved scalars
      parameter (nsessmax=1)           ! max sessions to NEKNEK
      parameter (lxo=lx1)              ! max GLL points on output (lxo>=
      parameter (mxprev=20)            ! max dim of projection space
      parameter (lgmres=30)            ! max dim Krylov space
      parameter (lorder=3)             ! max order in time
      parameter (lx1m=lx1)               ! GLL points mesh solver
      parameter (lfdm=0)               ! unused
      parameter (lelx=1,lely=1,lelz=1) ! global tensor mesh dimensions
      
      parameter (lbelt=1)              ! lelt for mhd
      parameter (lpelt=1)              ! lelt for linear stability
      parameter (lcvelt=1)             ! lelt for cvode
      
      ! INTERNALS

# 1 "/home/cmaloney111/Nek5000/core/SIZE.inc" 1
c - - SIZE internals
# 2
      integer lelv
      parameter(lelv=lelt)
      
      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))
      
      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))
      
      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))
      
      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))
      
      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))
      
      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))
      
      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)
      
      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))
      
      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))
      
      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)
      
      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))
      
      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))
      
      integer lxz
      parameter (lxz=lx1*lz1)
      
      integer lzl
      parameter (lzl=3 + 2*(ldim-3))
      
      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)
      
      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)
      
      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)
      
      integer maxmor
      parameter (maxmor = lelt)
      
      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control
      
      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Pre
      
      integer maxmbr
      parameter (maxmbr=lelt*6)
      
      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))
      
      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))
      
      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel
      
      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
# 1782 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1782 "/home/cmaloney111/Nek5000/core/drive2.f"

# 1 "/home/cmaloney111/Nek5000/core/TOTAL" 1

# 1 "/home/cmaloney111/Nek5000/core/DXYZ" 1
c     
c     Elemental derivative operators
c     
# 4
      real dxm1 (lx1,lx1), dxm12 (lx2,lx1)
     $   , dym1 (ly1,ly1), dym12 (ly2,ly1)
     $   , dzm1 (lz1,lz1), dzm12 (lz2,lz1)
     $   , dxtm1(lx1,lx1), dxtm12(lx1,lx2)
     $   , dytm1(ly1,ly1), dytm12(ly1,ly2)
     $   , dztm1(lz1,lz1), dztm12(lz1,lz2)
     $   , dxm3 (lx3,lx3), dxtm3 (lx3,lx3)
     $   , dym3 (ly3,ly3), dytm3 (ly3,ly3)
     $   , dzm3 (lz3,lz3), dztm3 (lz3,lz3)
     $   , dcm1 (ly1,ly1), dctm1 (ly1,ly1)
     $   , dcm3 (ly3,ly3), dctm3 (ly3,ly3)
     $   , dcm12(ly2,ly1), dctm12(ly1,ly2)
     $   , dam1 (ly1,ly1), datm1 (ly1,ly1)
     $   , dam12(ly2,ly1), datm12(ly1,ly2)
     $   , dam3 (ly3,ly3), datm3 (ly3,ly3)
      common /dxyz/ dxm1,dxm12,dym1,dym12,dzm1,dzm12,dxtm1,dxtm12,dytm1
     $             ,dytm12,dztm1,dztm12,dxm3,dxtm3,dym3,dytm3,dzm3
     $             ,dztm3,dcm1,dctm1,dcm3,dctm3,dcm12,dctm12,dam1,datm1
     $             ,dam12,datm12,dam3,datm3
# 2 "TOTAL" 2
# 2 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/DEALIAS" 1
c     
c    Dealiasing variables
c     
# 4
      real vxd(lxd,lyd,lzd,lelv)
     $   , vyd(lxd,lyd,lzd,lelv)
     $   , vzd(lxd,lyd,lzd,lelv)
      common /solnd/ vxd, vyd, vzd
      
      real imd1(lx1,lxd), imd1t(lxd,lx1)
     $   , im1d(lxd,lx1), im1dt(lx1,lxd)
     $   , pmd1(lx1,lxd), pmd1t(lxd,lx1)
      common /interpd/ imd1, imd1t, im1d, im1dt, pmd1, pmd1t
# 3 "TOTAL" 2
# 3 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/EIGEN" 1
c     
c     Eigenvalues
c     
# 4
      real eigas,eigaa,eigast,eigae,eigga,eiggs,eiggst,eigge
      common /eigval/ eigaa, eigas, eigast, eigae
     $               ,eigga, eiggs, eiggst, eigge
      
      logical         ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
      common /ifeig / ifaa,ifae,ifas,ifast,ifga,ifge,ifgs,ifgst
# 4 "TOTAL" 2
# 4 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/GEOM" 1
c     
c     Geometry arrays
c     
# 4
      real xm1(lx1,ly1,lz1,lelt)
     $    ,ym1(lx1,ly1,lz1,lelt)
     $    ,zm1(lx1,ly1,lz1,lelt)
     $    ,xm2(lx2,ly2,lz2,lelv)
     $    ,ym2(lx2,ly2,lz2,lelv)
     $    ,zm2(lx2,ly2,lz2,lelv)
      common /gxyz/ xm1,ym1,zm1,xm2,ym2,zm2
      
      real rxm1(lx1,ly1,lz1,lelt)
     $    ,sxm1(lx1,ly1,lz1,lelt)
     $    ,txm1(lx1,ly1,lz1,lelt)
     $    ,rym1(lx1,ly1,lz1,lelt)
     $    ,sym1(lx1,ly1,lz1,lelt)
     $    ,tym1(lx1,ly1,lz1,lelt)
     $    ,rzm1(lx1,ly1,lz1,lelt)
     $    ,szm1(lx1,ly1,lz1,lelt)
     $    ,tzm1(lx1,ly1,lz1,lelt)
     $    ,jacm1(lx1,ly1,lz1,lelt)
     $    ,jacmi(lx1*ly1*lz1,lelt)
      common /giso1/ rxm1,sxm1,txm1,rym1,sym1,tym1,rzm1,szm1,tzm1
     $              ,jacm1,jacmi
      
      real rxm2(lx2,ly2,lz2,lelv)
     $    ,sxm2(lx2,ly2,lz2,lelv)
     $    ,txm2(lx2,ly2,lz2,lelv)
     $    ,rym2(lx2,ly2,lz2,lelv)
     $    ,sym2(lx2,ly2,lz2,lelv)
     $    ,tym2(lx2,ly2,lz2,lelv)
     $    ,rzm2(lx2,ly2,lz2,lelv)
     $    ,szm2(lx2,ly2,lz2,lelv)
     $    ,tzm2(lx2,ly2,lz2,lelv)
     $    ,jacm2(lx2,ly2,lz2,lelv)
      common /giso2/ rxm2,sxm2,txm2,rym2,sym2,tym2,rzm2,szm2,tzm2
     $              ,jacm2
      
      real           rx(lxd*lyd*lzd,ldim*ldim,lelv)
      common /gisod/ rx
      
      real g1m1(lx1,ly1,lz1,lelt)
     $    ,g2m1(lx1,ly1,lz1,lelt)
     $    ,g3m1(lx1,ly1,lz1,lelt)
     $    ,g4m1(lx1,ly1,lz1,lelt)
     $    ,g5m1(lx1,ly1,lz1,lelt)
     $    ,g6m1(lx1,ly1,lz1,lelt)
      common /gmfact/ g1m1,g2m1,g3m1,g4m1,g5m1,g6m1
      
      real unr(lx1*lz1,6,lelt)
     $    ,uns(lx1*lz1,6,lelt)
     $    ,unt(lx1*lz1,6,lelt)
     $    ,unx(lx1,lz1,6,lelt)
     $    ,uny(lx1,lz1,6,lelt)
     $    ,unz(lx1,lz1,6,lelt)
     $    ,t1x(lx1,lz1,6,lelt)
     $    ,t1y(lx1,lz1,6,lelt)
     $    ,t1z(lx1,lz1,6,lelt)
     $    ,t2x(lx1,lz1,6,lelt)
     $    ,t2y(lx1,lz1,6,lelt)
     $    ,t2z(lx1,lz1,6,lelt)
     $    ,area(lx1,lz1,6,lelt)
     $    ,etalph(lx1*lz1,2*ldim,lelt)
     $    ,dlam
      common /gsurf/ unr,uns,unt,unx,uny,unz,t1x,t1y,t1z,t2x,t2y,t2z
     $             ,area,etalph,dlam
      
      real vnx(lx1m,ly1m,lz1m,lelt)
     $    ,vny(lx1m,ly1m,lz1m,lelt)
     $    ,vnz(lx1m,ly1m,lz1m,lelt)
     $    ,v1x(lx1m,ly1m,lz1m,lelt)
     $    ,v1y(lx1m,ly1m,lz1m,lelt)
     $    ,v1z(lx1m,ly1m,lz1m,lelt)
     $    ,v2x(lx1m,ly1m,lz1m,lelt)
     $    ,v2y(lx1m,ly1m,lz1m,lelt)
     $    ,v2z(lx1m,ly1m,lz1m,lelt)
      common /gvolm/ vnx,vny,vnz,v1x,v1y,v1z,v2x,v2y,v2z
      
      logical ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer(lelt),ifqinp(2*ldim,lelv),ifeppm(2*ldim,lelv)
     $       ,iflmsf(0:1),iflmse(0:1),iflmsc(0:1)
     $       ,ifmsfc(2*ldim,lelt,0:1)
     $       ,ifmseg(12,lelt,0:1)
     $       ,ifmscr(8,lelt,0:1)
     $       ,ifnskp(8,lelt)
     $       ,ifbcor
      common /glog/ ifgeom,ifgmsh3,ifvcor,ifsurt,ifmelt,ifwcno
     $       ,ifrzer,ifqinp,ifeppm
     $       ,iflmsf,iflmse,iflmsc,ifmsfc
     $       ,ifmseg,ifmscr,ifnskp
     $       ,ifbcor
      
      integer boundaryID(6,lelv), boundaryIDt(6,lelt)
      common /cbbid/ boundaryID, boundaryIDt
# 5 "TOTAL" 2
# 5 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/INPUT" 1
c     
c     Input parameters from preprocessors.
c     
c     Note that in parallel implementations, we distinguish between
c     distributed data (LELT) and uniformly distributed data.
c     
c     Input common block structure:
c     
c     INPUT1:  REAL            INPUT5: REAL      with LELT entries
c     INPUT2:  INTEGER         INPUT6: INTEGER   with LELT entries
c     INPUT3:  LOGICAL         INPUT7: LOGICAL   with LELT entries
c     INPUT4:  CHARACTER       INPUT8: CHARACTER with LELT entries
c     
# 14
      real param(200),rstim,vnekton
     $    ,cpfld(ldimt1,3)
     $    ,cpgrp(-5:10,ldimt1,3)
     $    ,qinteg(ldimt3,maxobj)
     $    ,uparam(20)
     $    ,atol(0:ldimt1)
     $    ,restol(0:ldimt1)
     $    ,fem_amg_param(15)
     $    ,crs_param(15)
     $    ,filterType
     $    ,connectivityTol
      
      common /input1/ param,rstim,vnekton,cpfld,cpgrp,qinteg,uparam,
     $                atol,restol,fem_amg_param,crs_param,
     $                filterType,connectivityTol
      
      integer matype(-5:10,ldimt1)
     $       ,nktonv,nhis,lochis(4,lhis+maxobj)
     $       ,ipscal,npscal,ipsco, ifldmhd
     $       ,irstv,irstt,irstim,nmember(maxobj),nobj
     $       ,ngeom,idpss(ldimt),fluid_partitioner,solid_partitioner
      common /input2/ matype,nktonv,nhis,lochis,ipscal,npscal,ipsco
     $               ,ifldmhd,irstv,irstt,irstim,nmember,nobj
     $               ,ngeom,idpss,fluid_partitioner,solid_partitioner
      
      logical         if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid
     $               ,ifadvc(ldimt1),ifdiff(ldimt1),ifdeal(ldimt1)
     $               ,iffilter(ldimt1),ifprojfld(0:ldimt1)
     $               ,iftmsh(0:ldimt1),ifdgfld(0:ldimt1),ifdg
     $               ,ifmvbd,ifchar,ifnonl(ldimt1)
     $               ,ifvarp(ldimt1),ifpsco(ldimt1),ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso(ldimt1),iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld(ldimt1),ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp
     $               ,ifbmap(ldimt1)
      
      common /input3/ if3d,ifflow,ifheat,iftran,ifaxis,ifstrs,ifsplit
     $               ,ifmgrid 
     $               ,ifadvc,ifdiff,ifdeal
     $               ,iffilter, ifprojfld
     $               ,iftmsh,ifdgfld,ifdg
     $               ,ifmvbd,ifchar,ifnonl
     $               ,ifvarp        ,ifpsco        ,ifvps
     $               ,ifmodel,ifkeps,ifintq,ifcons
     $               ,ifxyo,ifpo,ifvo,ifto,iftgo,ifpso        ,iffmtin
     $               ,ifbo,ifanls,ifanl2,ifmhd,ifessr,ifpert,ifbase
     $               ,ifcvode,iflomach,ifexplvis,ifschclob,ifuservp
     $               ,ifcyclic,ifmoab,ifcoup, ifvcoup, ifusermv,ifreguo
     $               ,ifxyo_,ifaziv,ifneknek,ifneknekm,ifneknekc
     $               ,ifcvfld,ifdp0dt
     $               ,ifmpiio,ifrich,ifvvisp,ifbmap
      
      logical         ifnav
      equivalence    (ifnav, ifadvc(1))
      
      character*1     hcode(11,lhis+maxobj)
      character*2     ocode(8)
      character*10    drivc(5)
      character*14    rstv,rstt
      character*40    textsw(100,2)
      character*132   initc(15)
      common /input4/ hcode,ocode,rstv,rstt,drivc,initc,textsw
      
      character*40    turbmod
      equivalence    (turbmod,textsw(1,1))
      
      character*132   reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      common /cfiles/ reafle,fldfle,dmpfle,hisfle,schfle,orefle,nrefle
      
      character*132   session,path,re2fle,parfle,amgfile
      common /cfile2/ session,path,re2fle,parfle,amgfile
      
      integer cr_re2,fh_re2
      common /handles_re2/ cr_re2,fh_re2
      
      integer*8 re2off_b
      common /off_re2/ re2off_b
c     
c proportional to LELT
c     
      real xc(8,lelt),yc(8,lelt),zc(8,lelt)
     $    ,bc(5,6,lelt,0:ldimt1)
     $    ,curve(6,12,lelt)
     $    ,cerror(lelt)
      common /input5/ xc,yc,zc,bc,curve,cerror
      
      integer igroup(lelt),object(maxobj,maxmbr,2)
      common /input6/ igroup,object
      
      integer lbid
      parameter(lbid = 100)
      
      character*1     ccurve(12,lelt),cdof(6,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
      character*3     cbc_bmap(lbid,ldimt1)
      integer         cbc_imap(lbid)
      integer         nbctype
      common /input8/ cbc,ccurve,cdof,cbc_bmap,cbc_imap,nbctype
      
      integer ieact(lelt),neact
      common /input9/ ieact,neact
c     
c material set ids, BC set ids, materials (f=fluid, s=solid), bc types
c     
      integer numsts
      parameter (numsts=50)
      
      integer numflu,numoth,numbcs 
     $       ,matindx(numsts),matids(numsts),imatie(lelt)
     $       ,ibcsts(numsts)
      common /inputmi/ numflu,numoth,numbcs,matindx,matids,imatie
     $                ,ibcsts
      
      integer bcf(numsts)
      common /inputmr/ bcf
      
      character*3 bctyps(numsts)
      common /inputmc/ bctyps
      
      integer out_mask(lelt)
      common /cbout_mask/ out_mask
# 6 "TOTAL" 2
# 6 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/IXYZ" 1
C     
C     Interpolation operators
C     
# 4
      real ixm12 (lx2,lx1),  ixm21 (lx1,lx2)
     $    ,iym12 (ly2,ly1),  iym21 (ly1,ly2)
     $    ,izm12 (lz2,lz1),  izm21 (lz1,lz2)
     $    ,ixtm12(lx1,lx2),  ixtm21(lx2,lx1)
     $    ,iytm12(ly1,ly2),  iytm21(ly2,ly1)
     $    ,iztm12(lz1,lz2),  iztm21(lz2,lz1)
     $    ,ixm13 (lx3,lx1),  ixm31 (lx1,lx3)
     $    ,iym13 (ly3,ly1),  iym31 (ly1,ly3)
     $    ,izm13 (lz3,lz1),  izm31 (lz1,lz3)
     $    ,ixtm13(lx1,lx3),  ixtm31(lx3,lx1)
     $    ,iytm13(ly1,ly3),  iytm31(ly3,ly1)
     $    ,iztm13(lz1,lz3),  iztm31(lz3,lz1)
      common /ixyz/ ixm12,iym12,izm12,ixm21,iym21,izm21
     $            , ixtm12,iytm12,iztm12,ixtm21,iytm21,iztm21
     $            , ixm13,iym13,izm13,ixm31,iym31,izm31
     $            , ixtm13,iytm13,iztm13,ixtm31,iytm31,iztm31
      
      real iam12 (ly2,ly1),  iam21 (ly1,ly2)
     $    ,iatm12(ly1,ly2),  iatm21(ly2,ly1)
     $    ,iam13 (ly3,ly1),  iam31 (ly1,ly3)
     $    ,iatm13(ly1,ly3),  iatm31(ly3,ly1)
     $    ,icm12 (ly2,ly1),  icm21 (ly1,ly2)
     $    ,ictm12(ly1,ly2),  ictm21(ly2,ly1)
     $    ,icm13 (ly3,ly1),  icm31 (ly1,ly3)
     $    ,ictm13(ly1,ly3),  ictm31(ly3,ly1)
     $    ,iajl1 (ly1,ly1),  iatjl1(ly1,ly1)
     $    ,iajl2 (ly2,ly2),  iatjl2(ly2,ly2)
     $    ,ialj3 (ly3,ly3),  iatlj3(ly3,ly3)
     $    ,ialj1 (ly1,ly1),  iatlj1(ly1,ly1)
      common /ixyza/ iam12,iam21,iatm12,iatm21,iam13,iam31,iatm13,iatm31
     $             , icm12,icm21,ictm12,ictm21,icm13,icm31,ictm13,ictm31
     $             , iajl1,iatjl1,iajl2,iatjl2,ialj3,iatlj3,ialj1,iatlj1
# 7 "TOTAL" 2
# 7 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MASS" 1
c     
c     Mass matrix
c     
# 4
      real bm1(lx1,ly1,lz1,lelt),bm2(lx2,ly2,lz2,lelv)
     $    ,binvm1(lx1,ly1,lz1,lelv),bintm1(lx1,ly1,lz1,lelt)
     $    ,bm2inv(lx2,ly2,lz2,lelt),baxm1(lx1,ly1,lz1,lelt)
     $    ,bm1lag(lx1,ly1,lz1,lelt,lorder-1)
     $    ,volvm1,volvm2,voltm1,voltm2
     $    ,yinvm1(lx1,ly1,lz1,lelt)
     $    ,binvdg(lx1*ly1*lz1,lelt)
     $    ,bm1ms(lx1,ly1,lz1,lelt)  !weighted mass matrix 
     $    ,upf(lx1,ly1,lz1,lelt)    !unity partition function
     $    ,volvm1ms
      common /mass/ bm1,bm2,binvm1,bintm1,bm2inv,baxm1,bm1lag
     $      ,volvm1,volvm2,voltm1,voltm2,yinvm1,binvdg
     $      ,bm1ms,upf,volvm1ms
# 8 "TOTAL" 2
# 8 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/MVGEOM" 1
C     
C     Moving mesh variables
C     
# 4
      real wx(lx1m,ly1m,lz1m,lelt)
     $   , wy(lx1m,ly1m,lz1m,lelt)
     $   , wz(lx1m,ly1m,lz1m,lelt)
      common /wsol/ wx,wy,wz
      
      real wxlag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wylag(lx1m,ly1m,lz1m,lelt,lorder-1)
     $   , wzlag(lx1m,ly1m,lz1m,lelt,lorder-1)
      common /wlag/ wxlag,wylag,wzlag
      
      real w1mask(lx1m,ly1m,lz1m,lelt)
     $   , w2mask(lx1m,ly1m,lz1m,lelt)
     $   , w3mask(lx1m,ly1m,lz1m,lelt)
     $   , wmult (lx1m,ly1m,lz1m,lelt)
      common /wmsu/ w1mask,w2mask,w3mask,wmult
      
      
      real ev1(lx1m,ly1m,lz1m,lelv)
     $   , ev2(lx1m,ly1m,lz1m,lelv)
     $   , ev3(lx1m,ly1m,lz1m,lelv)
      common /eigvec/ ev1,ev2,ev3
# 9 "TOTAL" 2
# 9 "TOTAL"

# 1 "/home/cmaloney111/TurbulentFlow/rans/rans_test/obj/PARALLEL" 1
c     
c     Communication information
c     NOTE: NID is stored in 'SIZE' for greater accessibility
# 4
      integer        node,pid,np,nullpid,node0
      common /cube1/ node,pid,np,nullpid,node0
c     
c     Maximum number of elements (limited to 2**31/12, at least for now)
      
      integer nelgt_max
      parameter(nelgt_max = 178956970)
      
      integer*8 nvtot
      integer nelg(0:ldimt1)
     $       ,lglel(lelt)
     $       ,gllel(lelg)
     $       ,gllnid(lelg)
     $       ,nelgv,nelgt
      common /hcglb/ nvtot,nelg,lglel,gllel,gllnid,nelgv,nelgt
      
      logical         ifgprnt
      common /diagl/  ifgprnt
      
      integer        wdsize,isize,isize8,lsize,csize,wdsizi
      common/precsn/ wdsize,isize,isize8,lsize,csize,wdsizi
      
      integer cr_h,gsh,gsh_fld(0:ldimt3),xxth(ldimt3)
      common /comm_handles/ cr_h,gsh,gsh_fld,xxth
      
      logical ifgsh_fld_same
      common /lcomm_handles/ ifgsh_fld_same
      
      integer              dg_face(lx1*lz1*2*ldim*lelt)
      common /xcdg_arrays/ dg_face
      
      integer            dg_hndlx,ndg_facex
      common /xcdg_ints/ dg_hndlx,ndg_facex
      
c     multisession
      integer nid_global, idsess_neighbor, intracomm, intercomm
     $      , iglobalcomm, npsess(0:nsessmax-1), np_neighbor, np_global
      common /nekmpi_global/ nid_global, idsess_neighbor
     $                     , intracomm, intercomm, iglobalcomm
     $                     , npsess,np_neighbor,np_global
      
      integer               nsessions
      common /session_info/ nsessions
# 10 "TOTAL" 2
# 10 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/SOLN" 1
c     
c     Main storage of simulation variables
c     
# 4
      integer lvt1,lvt2,lbt1,lbt2,lorder2
      parameter (lvt1  = lx1*ly1*lz1*lelv)
      parameter (lvt2  = lx2*ly2*lz2*lelv)
      parameter (lbt1  = lbx1*lby1*lbz1*lbelv)
      parameter (lbt2  = lbx2*lby2*lbz2*lbelv)
      
      parameter (lorder2 = max(1,lorder-2) )
c     
c     Solution and data
c     
      real bq(lx1,ly1,lz1,lelt,ldimt),adq(lx1,ly1,lz1,lelt,ldimt)
      common /bqcb/ bq,adq
      
c     Can be used for post-processing runs (SIZE .gt. 10+3*LDIMT flds)
      real vxlag  (lx1,ly1,lz1,lelv,2)
     $    ,vylag  (lx1,ly1,lz1,lelv,2)
     $    ,vzlag  (lx1,ly1,lz1,lelv,2)
     $    ,tlag   (lx1,ly1,lz1,lelt,lorder-1,ldimt)
     $    ,vgradt1(lx1,ly1,lz1,lelt,ldimt)
     $    ,vgradt2(lx1,ly1,lz1,lelt,ldimt)
     $    ,abx1   (lx1,ly1,lz1,lelv)
     $    ,aby1   (lx1,ly1,lz1,lelv)
     $    ,abz1   (lx1,ly1,lz1,lelv)
     $    ,abx2   (lx1,ly1,lz1,lelv)
     $    ,aby2   (lx1,ly1,lz1,lelv)
     $    ,abz2   (lx1,ly1,lz1,lelv)
     $    ,vdiff_e(lx1,ly1,lz1,lelt)
      
c     Solution data
      real vx     (lx1,ly1,lz1,lelv)
     $    ,vy     (lx1,ly1,lz1,lelv)
     $    ,vz     (lx1,ly1,lz1,lelv)
     $    ,vx_e   (lx1,ly1,lz1,lelv)
     $    ,vy_e   (lx1,ly1,lz1,lelv)
     $    ,vz_e   (lx1,ly1,lz1,lelv)
     $    ,t      (lx1,ly1,lz1,lelt,ldimt)
     $    ,vtrans (lx1,ly1,lz1,lelt,ldimt1)
     $    ,vdiff  (lx1,ly1,lz1,lelt,ldimt1)
     $    ,bfx    (lx1,ly1,lz1,lelv)
     $    ,bfy    (lx1,ly1,lz1,lelv)
     $    ,bfz    (lx1,ly1,lz1,lelv)
     $    ,cflf   (lx1,ly1,lz1,lelv)
     $    ,bmnv   (lx1*ly1*lz1*lelv*ldim,lorder+1) ! binv*mask
     $    ,bmass  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bmass
     $    ,bdivw  (lx1*ly1*lz1*lelv*ldim,lorder+1) ! bdivw*mask
     $    ,c_vx   (lxd*lyd*lzd*lelv*ldim,lorder+1) ! characteristics
     $    ,fw     (2*ldim,lelt)                    ! face weights for DG
      
      common /vptsol/ vxlag, vylag, vzlag, tlag, vgradt1, vgradt2,
     $     abx1, aby1, abz1, abx2, aby2, abz2, vdiff_e,
     $     vx, vy, vz, t, vtrans, vdiff, bfx, bfy, bfz, cflf, c_vx,fw,
     $     bmnv, bmass, bdivw,
     $     vx_e,vy_e,vz_e
      
c     Solution data for magnetic field
      real bx     (lbx1,lby1,lbz1,lbelv)
     $    ,by     (lbx1,lby1,lbz1,lbelv)
     $    ,bz     (lbx1,lby1,lbz1,lbelv)
     $    ,pm     (lbx2,lby2,lbz2,lbelv)
     $    ,bmx    (lbx1,lby1,lbz1,lbelv)  ! magnetic field rhs
     $    ,bmy    (lbx1,lby1,lbz1,lbelv)
     $    ,bmz    (lbx1,lby1,lbz1,lbelv)
     $    ,bbx1   (lbx1,lby1,lbz1,lbelv) ! extrapolation terms for
     $    ,bby1   (lbx1,lby1,lbz1,lbelv) ! magnetic field rhs
     $    ,bbz1   (lbx1,lby1,lbz1,lbelv)
     $    ,bbx2   (lbx1,lby1,lbz1,lbelv)
     $    ,bby2   (lbx1,lby1,lbz1,lbelv)
     $    ,bbz2   (lbx1,lby1,lbz1,lbelv)
     $    ,bxlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bylag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,bzlag  (lbx1*lby1*lbz1*lbelv,lorder-1)
     $    ,pmlag  (lbx2*lby2*lbz2*lbelv,lorder2)
      
      common /vptsolm/
     $     bx, by, bz, pm, bmx, bmy, bmz,
     $     bbx1, bby1, bbz1, bbx2, bby2, bbz2, bxlag, bylag, bzlag,
     $     pmlag
      
      real nu_star
      common /expvis/ nu_star
      
      real pr(lx2,ly2,lz2,lelv), prlag(lx2,ly2,lz2,lelv,lorder2)
      common /cbm2/ pr, prlag
      
      real qtl(lx2,ly2,lz2,lelt), usrdiv(lx2,ly2,lz2,lelt)
      common /diverg/ qtl, usrdiv
      
      real p0th, dp0thdt, gamma0, p0thn, p0thlag(2)
      common /p0therm/ p0th, dp0thdt, gamma0, p0thn, p0thlag
      
      real  v1mask (lx1,ly1,lz1,lelv)
     $     ,v2mask (lx1,ly1,lz1,lelv)
     $     ,v3mask (lx1,ly1,lz1,lelv)
     $     ,pmask  (lx1,ly1,lz1,lelv)
     $     ,tmask  (lx1,ly1,lz1,lelt,ldimt)
     $     ,omask  (lx1,ly1,lz1,lelt)
     $     ,vmult  (lx1,ly1,lz1,lelv)
     $     ,tmult  (lx1,ly1,lz1,lelt,ldimt)
     $     ,b1mask (lbx1,lby1,lbz1,lbelv)  ! masks for mag. field
     $     ,b2mask (lbx1,lby1,lbz1,lbelv)
     $     ,b3mask (lbx1,lby1,lbz1,lbelv)
     $     ,bpmask (lbx1,lby1,lbz1,lbelv)  ! magnetic pressure
      common /vptmsk/ v1mask,v2mask,v3mask,pmask,tmask,omask,vmult,
     $     tmult,b1mask,b2mask,b3mask,bpmask
c     
c     Solution and data for perturbation fields
c     
       real vxp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vyp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vzp    (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,prp    (lpx2*lpy2*lpz2*lpelv,lpert)
     $     ,tp     (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bqp    (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,adqp   (lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,bfxp   (lpx1*lpy1*lpz1*lpelv,lpert)  ! perturbation field rh
     $     ,bfyp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,bfzp   (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vxlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vylagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,vzlagp (lpx1*lpy1*lpz1*lpelv,lorder-1,lpert)
     $     ,prlagp (lpx2*lpy2*lpz2*lpelv,lorder2,lpert)
     $     ,tlagp  (lpx1*lpy1*lpz1*lpelt,ldimt,lorder-1,lpert)
     $     ,exx1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! extrapolation terms fo
     $     ,exy1p  (lpx1*lpy1*lpz1*lpelv,lpert) ! perturbation field rhs
     $     ,exz1p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exx2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exy2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,exz2p  (lpx1*lpy1*lpz1*lpelv,lpert)
     $     ,vgradt1p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
     $     ,vgradt2p(lpx1*lpy1*lpz1*lpelt,ldimt,lpert)
      common /pvptsl/ vxp, vyp, vzp, prp, tp, bqp, bfxp, bfyp, bfzp,
     $     vxlagp, vylagp, vzlagp, prlagp, tlagp,
     $     exx1p, exy1p, exz1p, exx2p, exy2p, exz2p,
     $     vgradt1p, vgradt2p, adqp
      
      integer jp
      common /ppointr/ jp
# 11 "TOTAL" 2
# 11 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/STEADY" 1
c     
c     Steady variables
c     
# 4
      real            tauss(ldimt1), txnext(ldimt1)
      common /sspar1/ tauss        , txnext
      
      integer nsskip
      common /sspar2/ nsskip
      
      logical         ifskip, ifmodp, ifssvt, ifstst(ldimt1)
     $              ,                 ifexvt, ifextr(ldimt1)
      common /sspar3/ ifskip, ifmodp, ifssvt, ifstst
     $              ,                 ifexvt, ifextr
      
      real dvnnh1,dvnnsm,dvnnl2,dvnnl8,dvdfh1,dvdfsm,
     $     dvdfl2,dvdfl8,dvprh1,dvprsm,dvprl2,dvprl8
      common /ssnorm/ dvnnh1, dvnnsm, dvnnl2, dvnnl8
     $              , dvdfh1, dvdfsm, dvdfl2, dvdfl8
     $              , dvprh1, dvprsm, dvprl2, dvprl8
# 12 "TOTAL" 2
# 12 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TOPOL" 1
c     
c     Arrays for direct stiffness summation
c     
# 4
      integer nomlis(2,3),nmlinv(6),group(6),skpdat(6,6),eface(6)
     $       ,eface1(6)
      common /cfaces/ nomlis,nmlinv,group,skpdat,eface,eface1
      
      integer eskip(-12:12,3),nedg(3),ncmp
     $       ,ixcn(8),noffst(3,0:ldimt1)
     $       ,maxmlt,nspmax(0:ldimt1)
     $       ,ngspcn(0:ldimt1),ngsped(3,0:ldimt1)
     $       ,numscn(lelt,0:ldimt1),numsed(lelt,0:ldimt1)
     $       ,gcnnum( 8,lelt,0:ldimt1),lcnnum( 8,lelt,0:ldimt1)
     $       ,gednum(12,lelt,0:ldimt1),lednum(12,lelt,0:ldimt1)
     $       ,gedtyp(12,lelt,0:ldimt1)
     $       ,ngcomm(2,0:ldimt1)
      common /cedges/ eskip,nedg,ncmp,ixcn,noffst,maxmlt,nspmax
     $               ,ngspcn,ngsped,numscn,numsed,gcnnum,lcnnum
     $               ,gednum,lednum,gedtyp,ngcomm
      
      integer iedge(20),iedgef(2,4,6,0:1)
     $       ,indx(8),invedg(27)
      common /edges/ iedge,iedgef,indx,invedg
      
      integer iedgfc(4,6)
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      
      integer icedg(3,16)
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      
      integer icface(4,10)
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C     
# 13 "TOTAL" 2
# 13 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/TSTEP" 1
c     
c     Variables related to time integration
c     
# 4
      real time,timef,fintim,timeio,timeioe
     $    ,dt,dtlag(10),dtinit,dtinvm,courno,ctarg
     $    ,ab(10),bd(10),abmsh(10)
     $    ,avdiff(ldimt1),avtran(ldimt1),volfld(0:ldimt1)
     $    ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $    ,tolps,tolhs,tolhr,tolhv,tolht(ldimt1),tolhe
     $    ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $    ,tnrmh1(ldimt),tnrmsm(ldimt),tnrml2(ldimt)
     $    ,tnrml8(ldimt),tmean(ldimt)
      common /tstep1/ time,timef,fintim,timeio,timeioe
     $               ,dt,dtlag,dtinit,dtinvm,courno,ctarg
     $               ,ab,bd,abmsh
     $               ,avdiff,avtran,volfld
     $               ,tolrel,tolabs,tolhdf,tolpdf,tolev,tolnl,prelax
     $               ,tolps,tolhs,tolhr,tolhv,tolht,tolhe
     $               ,vnrmh1,vnrmsm,vnrml2,vnrml8,vmean
     $               ,tnrmh1,tnrmsm,tnrml2
     $               ,tnrml8,tmean
      
      integer ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $       ,instep
     $       ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $       ,nmxt(ldimt),nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $       ,nelfld(0:ldimt1)
     $       ,nconv,nconv_max
     $       ,ioinfodmp
      common /istep2/ ifield,imesh,istep,nsteps,iostep,lastep,iocomm
     $               ,instep
     $               ,nab,nabmsh,nbd,nbdinp,ntaubd 
     $               ,nmxt,nmxh,nmxv,nmxp,nmxe,nmxnl,ninter
     $               ,nelfld
     $               ,nconv,nconv_max
     $               ,ioinfodmp
      
      real pi
      common /tstep3/ pi
      
      logical ifprnt,if_full_pres,ifoutfld
      common /tstep4/ ifprnt,if_full_pres,ifoutfld
      
      
      real lyap(3,lpert)
      common /tstep5/ lyap  !  lyapunov simulation history
# 14 "TOTAL" 2
# 14 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/ESOLV" 1
c     
c     Variables for E-solver
c     
# 4
      integer         iesolv
      common /econst/ iesolv
      
      logical         ifalgn(lelv), ifrsxy(lelv)
      common /efastm/ ifalgn      , ifrsxy
      
      real            volel(lelv)
      common /eouter/ volel       
# 15 "TOTAL" 2
# 15 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZ" 1
!     
!     Gauss-Labotto and Gauss points
!     
# 4
      real zgm1(lx1,3), zgm2(lx2,3), zgm3(lx3,3)
     $    ,zam1(lx1)  , zam2(lx2)  , zam3(lx3)
      common /gauss/ zgm1,zgm2,zgm3,zam1,zam2,zam3
!     
!    Weights
!     
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
     $    ,wxm2(lx2), wym2(ly2), wzm2(lz2), w3m2(lx2,ly2,lz2)
     $    ,wxm3(lx3), wym3(ly3), wzm3(lz3), w3m3(lx3,ly3,lz3)
     $    ,wam1(ly1), wam2(ly2), wam3(ly3)
     $    ,w2am1(lx1,ly1), w2cm1(lx1,ly1)
     $    ,w2am2(lx2,ly2), w2cm2(lx2,ly2)
     $    ,w2am3(lx3,ly3), w2cm3(lx3,ly3)
      common /wxyz/ wxm1,wym1,wzm1,w3m1,wxm2,wym2,wzm2,w3m2,wxm3,wym3
     $             ,wzm3,w3m3,wam1,wam2,wam3,w2am1,w2cm1,w2am2,w2cm2
     $             ,w2am3, w2cm3
# 16 "TOTAL" 2
# 16 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/WZF" 1
c     
c     Points (z) and weights (w) on velocity, pressure
c     
c     zgl -- velocity points on Gauss-Lobatto points i = 1,...nx
c     zgp -- pressure points on Gauss         points i = 1,...nxp (nxp =
c     
      
c     integer    lxm ! defined in HSMG
c     parameter (lxm = lx1)
# 10
      integer    lxq
      parameter (lxq = lx2)
c     
      real         zgl(lx1), wgl(lx1), zgp(lx1), wgp(lxq)
      common /wz1/ zgl     , wgl     , zgp     , wgp
c     
c     Tensor- (outer-) product of 1D weights   (for volumetric integrati
c     
      real         wgl1(lx1*lx1), wgl2(lxq*lxq), wgli(lx1*lx1)
      common /wz2/ wgl1         , wgl2         , wgli
c     
c     
c    Frequently used derivative matrices:
c     
c    D1, D1t   ---  differentiate on mesh 1 (velocity mesh)
c    D2, D2t   ---  differentiate on mesh 2 (pressure mesh)
c     
c    DXd,DXdt  ---  differentiate from velocity mesh ONTO dealiased mesh
c                   (currently the same as D1 and D1t...)
c     
c     
      real d1    (lx1*lx1) , d1t    (lx1*lx1)
     $   , d2    (lx1*lx1) , b2p    (lx1*lx1)
     $   , B1iA1 (lx1*lx1) , B1iA1t (lx1*lx1)
     $   , da    (lx1*lx1) , dat    (lx1*lx1)
     $   , iggl  (lx1*lxq) , igglt  (lx1*lxq)
     $   , dglg  (lx1*lxq) , dglgt  (lx1*lxq)
     $   , wglg  (lx1*lxq) , wglgt  (lx1*lxq)
      common /deriv/  d1,d1t,d2,b2p,B1iA1,B1iA1t
     $    ,da,dat,iggl,igglt,dglg,dglgt,wglg,wglgt
# 17 "TOTAL" 2
# 17 "TOTAL"

# 1 "/home/cmaloney111/Nek5000/core/OBJDATA" 1
# 1
      real dragx, dragpx, dragvx
      real dragy, dragpy, dragvy
      real dragz, dragpz, dragvz
      real torqx, torqpx, torqvx
      real torqy, torqpy, torqvy
      real torqz, torqpz, torqvz
      real dpdx_mean,dpdy_mean,dpdz_mean
      real dgtq 
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
# 1783 "/home/cmaloney111/Nek5000/core/drive2.f" 2
# 1783 "/home/cmaloney111/Nek5000/core/drive2.f"
      
      if(nio.eq.0) write(6,*) 'initialize pressure solver'
      isolver = param(40)
      
      if (isolver.eq.0) then      ! semg_xxt
         if (nelgt.gt.350000)
     $   call exitti('problem size too large for XXT solver!$',0)
         call set_overlap
      else if (isolver.eq.1) then ! semg_amg
         call set_overlap
      else if (isolver.eq.2) then ! semg_amg_hypre
         call set_overlap
      else if (isolver.eq.3) then ! fem_amg_hypre
         null_space = 0
         if (ifvcor) null_space = 1 
         call fem_amg_setup(nx1,ny1,nz1,
     $                      nelv,ndim,
     $                      xm1,ym1,zm1,
     $                      pmask,binvm1,null_space,
     $                      gsh_fld(1),fem_amg_param)
      endif
      
      return 
      end

# 1 "/home/cmaloney111/Nek5000/core/plan5.f"
c-----------------------------------------------------------------------
      subroutine plan5(igeom)
      
c     Two-step Richardson Extrapolation.
c     Operator splitting technique.
      

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
# 8 "/home/cmaloney111/Nek5000/core/plan5.f" 2
# 8 "/home/cmaloney111/Nek5000/core/plan5.f"

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
# 9 "/home/cmaloney111/Nek5000/core/plan5.f" 2
# 9 "/home/cmaloney111/Nek5000/core/plan5.f"
      
      common /scrns/  resv  (lx1*ly1*lz1*lelv,3)
      
      n   = lx1*ly1*lz1*nelv
      n2  = lx2*ly2*lz2*nelv
      dt2 = dt/2
      dti = 1/dt
      
      if (igeom.eq.2) then
      
      if (ifmvbd) call opcopy
     $  (wxlag(1,1,1,1,2),wylag(1,1,1,1,2),wzlag(1,1,1,1,2),xm1,ym1,zm1)
      
      do i=1,n
         s = bm1(i,1,1,1)*vtrans(i,1,1,1,1)*dti  ! Add  density*mass/dt,
         vxlag(i,1,1,1,2)=s*vx(i,1,1,1)          ! equivalent to using
         vylag(i,1,1,1,2)=s*vy(i,1,1,1)          ! density*mass/(dt/2)
         vzlag(i,1,1,1,2)=s*vz(i,1,1,1)          ! in the first place...
      enddo
      
      call midstep(vxlag,vylag,vzlag,prlag,0,dt)  ! One step of Pn-Pn-2
      
      do i=1,n                                          ! Add  density*m
         bfx(i,1,1,1)=bfx(i,1,1,1)+vxlag(i,1,1,1,2)     ! equivalent to 
         bfy(i,1,1,1)=bfy(i,1,1,1)+vylag(i,1,1,1,2)     ! density*mass/(
         bfz(i,1,1,1)=bfz(i,1,1,1)+vzlag(i,1,1,1,2)     ! in the first p
      enddo
      
      if (ifmvbd) then
        call opcopy
     $  (xm1,ym1,zm1,wxlag(1,1,1,1,2),wylag(1,1,1,1,2),wzlag(1,1,1,1,2))
        call geom_reset(0)
      endif
      
      time = time-dt2
      call midstep(vx,vy,vz,pr,1,dt2)      ! One step of Pn-Pn-2, dt/2
      
      time = time+dt2
      call setup_convect(2)  ! Map vx --> vxd
      call setprop
      
      call midstep(vx,vy,vz,pr,0,dt2)      ! One step of Pn-Pn-2, dt/2
      
      do i=1,n
         vx(i,1,1,1)=2*vx(i,1,1,1)-vxlag(i,1,1,1,1)
         vy(i,1,1,1)=2*vy(i,1,1,1)-vylag(i,1,1,1,1)
         vz(i,1,1,1)=2*vz(i,1,1,1)-vzlag(i,1,1,1,1)
      enddo
      
      do i=1,n2
         pr(i,1,1,1)=2*pr(i,1,1,1)-prlag(i,1,1,1,1)
      enddo
      
      call ortho(pr)
      
      endif
      
      return
      end
c-----------------------------------------------------------------------
      subroutine midstep(ux,uy,uz,pu,iresv,dtl)

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
# 71 "/home/cmaloney111/Nek5000/core/plan5.f" 2
# 71 "/home/cmaloney111/Nek5000/core/plan5.f"

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
# 72 "/home/cmaloney111/Nek5000/core/plan5.f" 2
# 72 "/home/cmaloney111/Nek5000/core/plan5.f"
      
      parameter (lv=lx1*ly1*lz1*lelt)
      real ux(1),uy(1),uz(1),pu(1)
      
      common /p5var/ rhs2   (lx1*ly1*lz1*lelv,3)
      
      common /scrns/  resv  (lx1*ly1*lz1*lelv,3)
     $ ,              dv1   (lx1*ly1*lz1*lelv)
     $ ,              dv2   (lx1*ly1*lz1*lelv)
     $ ,              dv3   (lx1*ly1*lz1*lelv)
      common /scrvh/  h1    (lx1*ly1*lz1*lelv)
     $ ,              h2    (lx1*ly1*lz1*lelv)
      
      
      if (lx1.eq.lx2) 
     $   call exitti('midstep requires lx2=lx1-2 in SIZE$',lx2)
      
      ifield = 1                ! Set field for velocity
      n   = lx1*ly1*lz1*nelv
      n2  = lx2*ly2*lz2*nelv
      
      dti = 1/dtl
      call copy    (h1,vdiff ,n)
      call cmult2  (h2,vtrans,dti,n)
      
      if (iresv.eq.0) then ! bfx etc is preserved if iresv=1
      
                    call makeuf  ! Initialize bfx, bfy, bfz
        if (ifmvbd) call admeshv ! Add div(W.u_i)
      
        call convop(resv(1,1),vx)
        call convop(resv(1,2),vy)
        call convop(resv(1,3),vz)
      
        do i=1,n
           b=vtrans(i,1,1,1,1)*bm1(i,1,1,1)
           s=1./dtl
           bfx(i,1,1,1)=bfx(i,1,1,1)+b*(s*vx(i,1,1,1)-resv(i,1))
           bfy(i,1,1,1)=bfy(i,1,1,1)+b*(s*vy(i,1,1,1)-resv(i,2))
           bfz(i,1,1,1)=bfz(i,1,1,1)+b*(s*vz(i,1,1,1)-resv(i,3))
        enddo
      
      endif
      
      call adv_geom(dtl) ! Advance the geometry
      
      call opcopy  (ux,uy,uz,vx,vy,vz)
      
      call bcdirvc (ux,uy,uz,v1mask,v2mask,v3mask) ! Don't forget bcneut
      call ophx    (resv(1,1),resv(1,2),resv(1,3),ux,uy,uz,h1,h2)
      
      call copy(rhs2,resv,lx1*ly1*lz1*lelv*3)
      
      do i=1,n
         resv(i,1)=bfx(i,1,1,1)-resv(i,1) ! rhs = rhs - H*u
         resv(i,2)=bfy(i,1,1,1)-resv(i,2)
         resv(i,3)=bfz(i,1,1,1)-resv(i,3)
      enddo
      
      tolhv = abs(param(22))
      call ophinv(dv1,dv2,dv3
     $   ,resv(1,1),resv(1,2),resv(1,3),h1,h2,tolhv,nmxv)
      
      call opadd2(ux,uy,uz,dv1,dv2,dv3)
      
      bd(1) = 1.0
      call rzero(pu,n2)
      
      dt_old = dt
      dt = dtl
      call incomprn(ux,uy,uz,pu)
      dt = dt_old
      
      return
      end
c-----------------------------------------------------------------------
      subroutine adv_geom(dtl) ! Advance the geometry

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
# 150 "/home/cmaloney111/Nek5000/core/plan5.f" 2
# 150 "/home/cmaloney111/Nek5000/core/plan5.f"

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
# 151 "/home/cmaloney111/Nek5000/core/plan5.f" 2
# 151 "/home/cmaloney111/Nek5000/core/plan5.f"
      
      param(28) = 1   !  This forces Euler Forward for Mesh Update
                      !  Note: p28 must be set prior to call settime
      
      dt_tmp = dt     ! Save "full" dt value
      dt     = dtl
      
      ifield = 1
      call gengeom(2)
      ifield = 1
      
      dt     = dt_tmp ! Restore dt
      
      return
      end
c-----------------------------------------------------------------------

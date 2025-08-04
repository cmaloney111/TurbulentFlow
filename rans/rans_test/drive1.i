# 1 "/home/cmaloney111/Nek5000/core/drive1.f"
c-----------------------------------------------------------------------
      subroutine nek_init(comm)
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
# 5 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 5 "/home/cmaloney111/Nek5000/core/drive1.f"

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
# 6 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 6 "/home/cmaloney111/Nek5000/core/drive1.f"

# 1 "/home/cmaloney111/Nek5000/core/DOMAIN" 1
c     
c     Arrays for overlapping Schwartz algorithm
c     
# 4
      integer ltotd
      parameter (ltotd = lx1*ly1*lz1*lelt                     )
c     
      integer ndom, n_o, nel_proc, gs_hnd_overlap
     $      , na (lelt+1) , ma(lelt+1), nza(lelt+1)
      common /ddptri/ ndom,n_o,nel_proc,gs_hnd_overlap,na,ma,nza
c     
c     These are the H1 coarse-grid arrays:
c     
      integer lxc, lcr
      parameter(lxc=2)
      parameter(lcr=lxc**ldim)
      
      integer*8 se_to_gcrs(lcr,lelt)
      integer n_crs,m_crs,nx_crs,nxyz_c
      common /h1_crsi/ se_to_gcrs, n_crs,m_crs, nx_crs, nxyz_c
c     
      real             h1_basis(lx1*lxc), h1_basist(lxc*lx1)
      common /h1_crs/  h1_basis         , h1_basist
      
      real             l2_basis(lx2*lxc), l2_basist(lxc*lx2)
      equivalence     (h1_basis  , l2_basis  )
      equivalence     (h1_basist , l2_basist )
# 7 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 7 "/home/cmaloney111/Nek5000/core/drive1.f"
c     

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
# 9 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 9 "/home/cmaloney111/Nek5000/core/drive1.f"

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
# 10 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 10 "/home/cmaloney111/Nek5000/core/drive1.f"
      
C     used scratch arrays
C     NOTE: no initial declaration needed. Linker will take 
c           care about the size of the CBs automatically
c     
c      COMMON /CTMP1/ DUMMY1(LCTMP1)
c      COMMON /CTMP0/ DUMMY0(LCTMP0)
c     
c      COMMON /SCRNS/ DUMMY2(LX1,LY1,LZ1,LELT,7)
c      COMMON /SCRUZ/ DUMMY3(LX1,LY1,LZ1,LELT,4)
c      COMMON /SCREV/ DUMMY4(LX1,LY1,LZ1,LELT,2)
c      COMMON /SCRVH/ DUMMY5(LX1,LY1,LZ1,LELT,2)
c      COMMON /SCRMG/ DUMMY6(LX1,LY1,LZ1,LELT,4)
c      COMMON /SCRCH/ DUMMY7(LX1,LY1,LZ1,LELT,2)
c      COMMON /SCRSF/ DUMMY8(LX1,LY1,LZ1,LELT,3)
c      COMMON /SCRCG/ DUMM10(LX1,LY1,LZ1,LELT,1)
      
      integer comm
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
      
      common /rdump/ ntdump
      
      real kwave2
      logical ifemati
      
      real rtest
      integer itest
      integer*8 itest8
      character ctest
      logical ltest 
      
      common /c_is1/ glo_num(lx1 * ly1 * lz1, lelt)
      common /ivrtx/ vertex((2 ** ldim) * lelt)
      integer*8 glo_num, ngv
      integer*8 vertex
      
      ! set word size for REAL
      wdsize = sizeof(rtest)
      ! set word size for INTEGER
      isize = sizeof(itest)
      ! set word size for INTEGER*8
      isize8 = sizeof(itest8) 
      ! set word size for LOGICAL
      lsize = sizeof(ltest) 
      ! set word size for CHARACTER
      csize = sizeof(ctest)
      
      call setupcomm(comm,newcomm,newcommg,'','')
      intracomm   = newcomm   ! within a session
      nekcomm     = newcomm
      iglobalcomm = newcommg  ! across all sessions
      call iniproc()
      
      if (nid.eq.nio) call printHeader
      
      etimes = dnekclock()
      istep  = 0
      
      call opcount(1)
      
      call initdim         ! Initialize / set default values.
      call initdat
      call files
      
      call readat          ! Read .rea +map file
      
      if (nio.eq.0) then
         write(6,12) 'nelgt/nelgv/lelt:',nelgt,nelgv,lelt
         write(6,12) 'lx1/lx2/lx3/lxd: ',lx1,lx2,lx3,lxd
 12      format(1X,A,4I12)
         write(6,*)
      endif
      
      call setvar          ! Initialize most variables
      
      instep=1             ! Check for zero steps
      if (nsteps.eq.0 .and. fintim.eq.0.) instep=0
      
      igeom = 2
      call setup_topo      ! Setup domain topology  
      
      call genwz           ! Compute GLL points, weights, etc.
      
      if(nio.eq.0) write(6,*) 'call usrdat'
      call usrdat
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat' 
      
      call gengeom(igeom)  ! Generate geometry, after usrdat 
      
      if (ifmvbd) call setup_mesh_dssum ! Set mesh dssum (needs geom)
      
      if(nio.eq.0) write(6,*) 'call usrdat2'
      call usrdat2
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat2' 
      
      call count_bdry   ! count the number of faces with assigned BCs
      call fix_geom
      
      call vrdsmsh          ! verify mesh topology
      call mesh_metrics     ! print some metrics
      
      call setlog(.true.)   ! Initalize logical flags
      
      if (ifneknekc) call neknek_setup
      
      call bcmask  ! Set BC masks for Dirichlet boundaries.
      
      if (fintim.ne.0.0 .or. nsteps.ne.0) 
     $   call geneig(igeom) ! eigvals for tolerances
      
      call dg_setup ! Setup DG, if dg flag is set.
      
      if (ifflow.and.iftran) then ! Init pressure solver 
         if (fintim.ne.0 .or. nsteps.ne.0) call prinit
      endif
      
      if(ifcvode) call cv_setsize
      
      if(nio.eq.0) write(6,*) 'call usrdat3'
      call usrdat3
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat3'
      
      call setics
      call setprop
      
      if (instep.ne.0) then
         if (ifneknekc) call neknek_exchange
         if (ifneknekc) call chk_outflow
      
         if (nio.eq.0) write(6,*) 'call userchk'
         call userchk
         if(nio.eq.0) write(6,'(A,/)') ' done :: userchk' 
      endif
      
      call setprop      ! call again because input has changed in userch
      
      if (ifcvode .and. nsteps.gt.0) call cv_init
      
      call comment
      call sstest (isss) 
      
      call dofcnt
      
      jp = 0  ! Set perturbation field count to 0 for baseline flow
      p0thn = p0th
      
      call in_situ_init()
      
      call time00       !     Initalize timers to ZERO
      call opcount(2)
      
      ntdump=0
      if (timeio.ne.0.0) ntdump = int( time/timeio )
      
      tinit = dnekclock_sync() - etimes
      if (nio.eq.0) then
        write (6,*) ' '
        if (time.ne.0.0) write (6,'(a,e14.7)') ' Initial time:',time
        write (6,'(a,g13.5,a)') 
     &     ' Initialization successfully completed ', tinit, ' sec'
      endif
      
      return
      end
c-----------------------------------------------------------------------
      subroutine nek_solve
      

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
# 178 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 178 "/home/cmaloney111/Nek5000/core/drive1.f"

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
# 179 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 179 "/home/cmaloney111/Nek5000/core/drive1.f"

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
# 180 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 180 "/home/cmaloney111/Nek5000/core/drive1.f"

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
# 181 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 181 "/home/cmaloney111/Nek5000/core/drive1.f"
      
      call nekgsync()
      
      if (instep.eq.0) then
        if(nid.eq.0) write(6,'(/,A,/,A,/)') 
     &     ' nsteps=0 -> skip time loop',
     &     ' running solver in post processing mode'
      else
        if(nio.eq.0) write(6,'(/,A,/)') 'Starting time loop ...'
      endif
      
      isyc  = 0
      if(ifsync) isyc=1
      itime = 0

# 196
      itime = 1

      
      ! start measurements
# 200
      dtmp = dnekgflops()
      
      istep  = 0
      msteps = 1
      
      irstat = int(param(120))
      
      do kstep=1,nsteps,msteps
         call nek__multi_advance(kstep,msteps)
         if(kstep.ge.nsteps) lastep = 1
         call check_ioinfo  
         call set_outfld
         etime1 = dnekclock()
         call userchk
         tuchk = tuchk + dnekclock()-etime1
         call prepost (ifoutfld,'his')
         call in_situ_check()
         if (mod(kstep,irstat).eq.0 .and. lastep.eq.0) call runstat 
         if (lastep .eq. 1) goto 1001
      enddo
 1001 lastep=1
      
      call comment
      
c     check for post-processing mode
      if (instep.eq.0) then
         nsteps=0
         istep=0
         if(nio.eq.0) write(6,*) 'call userchk'
         call userchk
         if(nio.eq.0) write(6,*) 'done :: userchk'
         call prepost (.true.,'his')
      else
         if (nio.eq.0) write(6,'(/,A,/)') 
     $      'end of time-step loop' 
      endif
      
      
      RETURN
      END
      
c-----------------------------------------------------------------------
      subroutine nek_advance
      

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
# 245 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 245 "/home/cmaloney111/Nek5000/core/drive1.f"

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
# 246 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 246 "/home/cmaloney111/Nek5000/core/drive1.f"

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
# 247 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 247 "/home/cmaloney111/Nek5000/core/drive1.f"
      
      common /cgeom/ igeom
      
      ntot = lx1*ly1*lz1*nelv
      
      call nekgsync
      
      call setup_convect(2) ! Save conv vel
      
      if (iftran) call settime
      if (ifmhd ) call cfl_check
      call setsolv
      call comment
      
      if (ifsplit) then   ! PN/PN formulation
      
         do igeom=1,ngeom
      
         if (ifneknekc .and. igeom.gt.2) then
            if (ifneknekm.and.igeom.eq.3) call neknek_setup
            call neknek_exchange
         endif
      
         ! call here before we overwrite wx 
         if (ifheat .and. ifcvode) call heat_cvode (igeom)   
      
         if (ifgeom) then
            call gengeom (igeom)
            call geneig  (igeom)
         endif
      
         if (ifheat) call heat (igeom)
      
         if (igeom.eq.2) then  
            call setprop
            call rzero(qtl,ntot)
            if (iflomach) call qthermal
         endif
      
         if (ifflow)          call fluid    (igeom)
         if (ifmvbd)          call meshv    (igeom)
         if (igeom.eq.ngeom.and.filterType.eq.1)
     $                        call q_filter(param(103))
      
         enddo
      
      else                ! PN-2/PN-2 formulation
         call setprop
         do igeom=1,ngeom
      
            if (ifneknekc .and. igeom.gt.2) then
              if (ifneknekm.and.igeom.eq.3) call neknek_setup
              call neknek_exchange
            endif
      
            ! call here before we overwrite wx 
            if (ifheat .and. ifcvode) call heat_cvode (igeom)   
      
            if (ifgeom) then
               if (.not.ifrich) call gengeom (igeom)
               call geneig  (igeom)
            endif
      
            if (ifmhd) then
               if (ifheat)      call heat     (igeom)
                                call induct   (igeom)
            elseif (ifpert) then
               if (ifbase.and.ifheat)  call heat          (igeom)
               if (ifbase.and.ifflow)  call fluid         (igeom)
               if (ifflow)             call fluidp        (igeom)
               if (ifheat)             call heatp         (igeom)
            else  ! std. nek case
               if (ifheat)             call heat          (igeom)
               if (ifflow)             call fluid         (igeom)
               if (ifmvbd)             call meshv         (igeom)
            endif
            if (igeom.eq.ngeom.and.filterType.eq.1)
     $         call q_filter(param(103))
         enddo
      endif
      
      return
      end
      
c-----------------------------------------------------------------------
      subroutine nek_end
      

# 1 "/opt/cray/pe/mpich/8.1.29/ofi/cray/17.0/include/mpif.h" 1
!      /* -*- Mode: Fortran; -*- */
!      
!      (C) 2001 by Argonne National Laboratory.
!      See COPYRIGHT in top-level directory.
!      
!      DO NOT EDIT
!      This file created by buildiface 
!      
# 9
       INTEGER MPI_SOURCE, MPI_TAG, MPI_ERROR
       PARAMETER (MPI_SOURCE=3,MPI_TAG=4,MPI_ERROR=5)
       INTEGER MPI_STATUS_SIZE
       PARAMETER (MPI_STATUS_SIZE=5)
       INTEGER MPI_STATUS_IGNORE(MPI_STATUS_SIZE)
       INTEGER MPI_STATUSES_IGNORE(MPI_STATUS_SIZE,1)
       INTEGER MPI_ERRCODES_IGNORE(1)
       CHARACTER*1 MPI_ARGVS_NULL(1,1)
       CHARACTER*1 MPI_ARGV_NULL(1)
       INTEGER MPI_SUCCESS
       PARAMETER (MPI_SUCCESS=0)
       INTEGER MPI_ERR_NO_SPACE
       PARAMETER (MPI_ERR_NO_SPACE=36)
       INTEGER MPI_ERR_TAG
       PARAMETER (MPI_ERR_TAG=4)
       INTEGER MPI_ERR_INFO_NOKEY
       PARAMETER (MPI_ERR_INFO_NOKEY=31)
       INTEGER MPI_ERR_OTHER
       PARAMETER (MPI_ERR_OTHER=15)
       INTEGER MPI_ERR_DISP
       PARAMETER (MPI_ERR_DISP=52)
       INTEGER MPI_ERR_NAME
       PARAMETER (MPI_ERR_NAME=33)
       INTEGER MPI_ERR_LOCKTYPE
       PARAMETER (MPI_ERR_LOCKTYPE=47)
       INTEGER MPI_ERR_READ_ONLY
       PARAMETER (MPI_ERR_READ_ONLY=40)
       INTEGER MPI_ERR_COMM
       PARAMETER (MPI_ERR_COMM=5)
       INTEGER MPI_ERR_UNSUPPORTED_DATAREP
       PARAMETER (MPI_ERR_UNSUPPORTED_DATAREP=43)
       INTEGER MPI_ERR_RMA_ATTACH
       PARAMETER (MPI_ERR_RMA_ATTACH=56)
       INTEGER MPI_ERR_INTERN
       PARAMETER (MPI_ERR_INTERN=16)
       INTEGER MPI_ERR_INFO_VALUE
       PARAMETER (MPI_ERR_INFO_VALUE=30)
       INTEGER MPI_ERR_RMA_FLAVOR
       PARAMETER (MPI_ERR_RMA_FLAVOR=58)
       INTEGER MPI_ERR_ARG
       PARAMETER (MPI_ERR_ARG=12)
       INTEGER MPI_ERR_RMA_SHARED
       PARAMETER (MPI_ERR_RMA_SHARED=57)
       INTEGER MPI_ERR_FILE_IN_USE
       PARAMETER (MPI_ERR_FILE_IN_USE=26)
       INTEGER MPI_ERR_QUOTA
       PARAMETER (MPI_ERR_QUOTA=39)
       INTEGER MPI_ERR_TYPE
       PARAMETER (MPI_ERR_TYPE=3)
       INTEGER MPI_ERR_UNSUPPORTED_OPERATION
       PARAMETER (MPI_ERR_UNSUPPORTED_OPERATION=44)
       INTEGER MPI_ERR_ASSERT
       PARAMETER (MPI_ERR_ASSERT=53)
       INTEGER MPI_ERR_INFO_KEY
       PARAMETER (MPI_ERR_INFO_KEY=29)
       INTEGER MPI_ERR_PORT
       PARAMETER (MPI_ERR_PORT=38)
       INTEGER MPI_ERR_TRUNCATE
       PARAMETER (MPI_ERR_TRUNCATE=14)
       INTEGER MPI_ERR_GROUP
       PARAMETER (MPI_ERR_GROUP=8)
       INTEGER MPI_ERR_WIN
       PARAMETER (MPI_ERR_WIN=45)
       INTEGER MPI_ERR_NOT_SAME
       PARAMETER (MPI_ERR_NOT_SAME=35)
       INTEGER MPI_ERR_NO_SUCH_FILE
       PARAMETER (MPI_ERR_NO_SUCH_FILE=37)
       INTEGER MPI_ERR_ROOT
       PARAMETER (MPI_ERR_ROOT=7)
       INTEGER MPI_ERR_RMA_CONFLICT
       PARAMETER (MPI_ERR_RMA_CONFLICT=49)
       INTEGER MPI_ERR_LASTCODE
       PARAMETER (MPI_ERR_LASTCODE=1073741823)
       INTEGER MPI_ERR_REQUEST
       PARAMETER (MPI_ERR_REQUEST=19)
       INTEGER MPI_ERR_IO
       PARAMETER (MPI_ERR_IO=32)
       INTEGER MPI_ERR_ACCESS
       PARAMETER (MPI_ERR_ACCESS=20)
       INTEGER MPI_ERR_FILE_EXISTS
       PARAMETER (MPI_ERR_FILE_EXISTS=25)
       INTEGER MPI_ERR_NO_MEM
       PARAMETER (MPI_ERR_NO_MEM=34)
       INTEGER MPI_ERR_UNKNOWN
       PARAMETER (MPI_ERR_UNKNOWN=13)
       INTEGER MPI_ERR_BAD_FILE
       PARAMETER (MPI_ERR_BAD_FILE=22)
       INTEGER MPI_ERR_COUNT
       PARAMETER (MPI_ERR_COUNT=2)
       INTEGER MPI_ERR_IN_STATUS
       PARAMETER (MPI_ERR_IN_STATUS=17)
       INTEGER MPI_ERR_SERVICE
       PARAMETER (MPI_ERR_SERVICE=41)
       INTEGER MPI_ERR_OP
       PARAMETER (MPI_ERR_OP=9)
       INTEGER MPI_ERR_TOPOLOGY
       PARAMETER (MPI_ERR_TOPOLOGY=10)
       INTEGER MPI_ERR_RMA_SYNC
       PARAMETER (MPI_ERR_RMA_SYNC=50)
       INTEGER MPI_ERR_SPAWN
       PARAMETER (MPI_ERR_SPAWN=42)
       INTEGER MPI_ERR_FILE
       PARAMETER (MPI_ERR_FILE=27)
       INTEGER MPI_ERR_SIZE
       PARAMETER (MPI_ERR_SIZE=51)
       INTEGER MPI_ERR_RMA_RANGE
       PARAMETER (MPI_ERR_RMA_RANGE=55)
       INTEGER MPI_ERR_INFO
       PARAMETER (MPI_ERR_INFO=28)
       INTEGER MPI_ERR_BASE
       PARAMETER (MPI_ERR_BASE=46)
       INTEGER MPI_ERR_BUFFER
       PARAMETER (MPI_ERR_BUFFER=1)
       INTEGER MPI_ERR_RANK
       PARAMETER (MPI_ERR_RANK=6)
       INTEGER MPI_ERR_DIMS
       PARAMETER (MPI_ERR_DIMS=11)
       INTEGER MPI_ERR_AMODE
       PARAMETER (MPI_ERR_AMODE=21)
       INTEGER MPI_ERR_PENDING
       PARAMETER (MPI_ERR_PENDING=18)
       INTEGER MPI_ERR_KEYVAL
       PARAMETER (MPI_ERR_KEYVAL=48)
       INTEGER MPI_ERR_DUP_DATAREP
       PARAMETER (MPI_ERR_DUP_DATAREP=24)
       INTEGER MPI_ERR_CONVERSION
       PARAMETER (MPI_ERR_CONVERSION=23)
       INTEGER MPI_ERRORS_ARE_FATAL
       PARAMETER (MPI_ERRORS_ARE_FATAL=1409286144)
       INTEGER MPI_ERRORS_RETURN
       PARAMETER (MPI_ERRORS_RETURN=1409286145)
       INTEGER MPI_IDENT
       PARAMETER (MPI_IDENT=0)
       INTEGER MPI_CONGRUENT
       PARAMETER (MPI_CONGRUENT=1)
       INTEGER MPI_SIMILAR
       PARAMETER (MPI_SIMILAR=2)
       INTEGER MPI_UNEQUAL
       PARAMETER (MPI_UNEQUAL=3)
       INTEGER MPI_WIN_FLAVOR_CREATE
       PARAMETER (MPI_WIN_FLAVOR_CREATE=1)
       INTEGER MPI_WIN_FLAVOR_ALLOCATE
       PARAMETER (MPI_WIN_FLAVOR_ALLOCATE=2)
       INTEGER MPI_WIN_FLAVOR_DYNAMIC
       PARAMETER (MPI_WIN_FLAVOR_DYNAMIC=3)
       INTEGER MPI_WIN_FLAVOR_SHARED
       PARAMETER (MPI_WIN_FLAVOR_SHARED=4)
       INTEGER MPI_WIN_SEPARATE
       PARAMETER (MPI_WIN_SEPARATE=1)
       INTEGER MPI_WIN_UNIFIED
       PARAMETER (MPI_WIN_UNIFIED=2)
       INTEGER MPI_MAX
       PARAMETER (MPI_MAX=1476395009)
       INTEGER MPI_MIN
       PARAMETER (MPI_MIN=1476395010)
       INTEGER MPI_SUM
       PARAMETER (MPI_SUM=1476395011)
       INTEGER MPI_PROD
       PARAMETER (MPI_PROD=1476395012)
       INTEGER MPI_LAND
       PARAMETER (MPI_LAND=1476395013)
       INTEGER MPI_BAND
       PARAMETER (MPI_BAND=1476395014)
       INTEGER MPI_LOR
       PARAMETER (MPI_LOR=1476395015)
       INTEGER MPI_BOR
       PARAMETER (MPI_BOR=1476395016)
       INTEGER MPI_LXOR
       PARAMETER (MPI_LXOR=1476395017)
       INTEGER MPI_BXOR
       PARAMETER (MPI_BXOR=1476395018)
       INTEGER MPI_MINLOC
       PARAMETER (MPI_MINLOC=1476395019)
       INTEGER MPI_MAXLOC
       PARAMETER (MPI_MAXLOC=1476395020)
       INTEGER MPI_REPLACE
       PARAMETER (MPI_REPLACE=1476395021)
       INTEGER MPI_NO_OP
       PARAMETER (MPI_NO_OP=1476395022)
       INTEGER MPI_COMM_WORLD
       PARAMETER (MPI_COMM_WORLD=1140850688)
       INTEGER MPI_COMM_SELF
       PARAMETER (MPI_COMM_SELF=1140850689)
       INTEGER MPI_GROUP_EMPTY
       PARAMETER (MPI_GROUP_EMPTY=1207959552)
       INTEGER MPI_COMM_NULL
       PARAMETER (MPI_COMM_NULL=67108864)
       INTEGER MPI_WIN_NULL
       PARAMETER (MPI_WIN_NULL=536870912)
       INTEGER MPI_FILE_NULL
       PARAMETER (MPI_FILE_NULL=0)
       INTEGER MPI_GROUP_NULL
       PARAMETER (MPI_GROUP_NULL=134217728)
       INTEGER MPI_OP_NULL
       PARAMETER (MPI_OP_NULL=402653184)
       INTEGER MPI_DATATYPE_NULL
       PARAMETER (MPI_DATATYPE_NULL=201326592)
       INTEGER MPI_REQUEST_NULL
       PARAMETER (MPI_REQUEST_NULL=738197504)
       INTEGER MPI_ERRHANDLER_NULL
       PARAMETER (MPI_ERRHANDLER_NULL=335544320)
       INTEGER MPI_INFO_NULL
       PARAMETER (MPI_INFO_NULL=469762048)
       INTEGER MPI_INFO_ENV
       PARAMETER (MPI_INFO_ENV=1543503873)
       INTEGER MPI_TAG_UB
       PARAMETER (MPI_TAG_UB=1681915906)
       INTEGER MPI_HOST
       PARAMETER (MPI_HOST=1681915908)
       INTEGER MPI_IO
       PARAMETER (MPI_IO=1681915910)
       INTEGER MPI_WTIME_IS_GLOBAL
       PARAMETER (MPI_WTIME_IS_GLOBAL=1681915912)
       INTEGER MPI_UNIVERSE_SIZE
       PARAMETER (MPI_UNIVERSE_SIZE=1681915914)
       INTEGER MPI_LASTUSEDCODE
       PARAMETER (MPI_LASTUSEDCODE=1681915916)
       INTEGER MPI_APPNUM
       PARAMETER (MPI_APPNUM=1681915918)
       INTEGER MPI_WIN_BASE
       PARAMETER (MPI_WIN_BASE=1711276034)
       INTEGER MPI_WIN_SIZE
       PARAMETER (MPI_WIN_SIZE=1711276036)
       INTEGER MPI_WIN_DISP_UNIT
       PARAMETER (MPI_WIN_DISP_UNIT=1711276038)
       INTEGER MPI_WIN_CREATE_FLAVOR
       PARAMETER (MPI_WIN_CREATE_FLAVOR=1711276040)
       INTEGER MPI_WIN_MODEL
       PARAMETER (MPI_WIN_MODEL=1711276042)
       INTEGER MPI_MAX_ERROR_STRING
       PARAMETER (MPI_MAX_ERROR_STRING=512-1)
       INTEGER MPI_MAX_PORT_NAME
       PARAMETER (MPI_MAX_PORT_NAME=255)
       INTEGER MPI_MAX_OBJECT_NAME
       PARAMETER (MPI_MAX_OBJECT_NAME=127)
       INTEGER MPI_MAX_INFO_KEY
       PARAMETER (MPI_MAX_INFO_KEY=254)
       INTEGER MPI_MAX_INFO_VAL
       PARAMETER (MPI_MAX_INFO_VAL=1023)
       INTEGER MPI_MAX_PROCESSOR_NAME
       PARAMETER (MPI_MAX_PROCESSOR_NAME=128-1)
       INTEGER MPI_MAX_DATAREP_STRING
       PARAMETER (MPI_MAX_DATAREP_STRING=127)
       INTEGER MPI_MAX_LIBRARY_VERSION_STRING
       PARAMETER (MPI_MAX_LIBRARY_VERSION_STRING=8192-1)
       INTEGER MPI_UNDEFINED
       PARAMETER (MPI_UNDEFINED=(-32766))
       INTEGER MPI_KEYVAL_INVALID
       PARAMETER (MPI_KEYVAL_INVALID=603979776)
       INTEGER MPI_BSEND_OVERHEAD
       PARAMETER (MPI_BSEND_OVERHEAD=96)
       INTEGER MPI_PROC_NULL
       PARAMETER (MPI_PROC_NULL=-1)
       INTEGER MPI_ANY_SOURCE
       PARAMETER (MPI_ANY_SOURCE=-2)
       INTEGER MPI_ANY_TAG
       PARAMETER (MPI_ANY_TAG=-1)
       INTEGER MPI_ROOT
       PARAMETER (MPI_ROOT=-3)
       INTEGER MPI_GRAPH
       PARAMETER (MPI_GRAPH=1)
       INTEGER MPI_CART
       PARAMETER (MPI_CART=2)
       INTEGER MPI_DIST_GRAPH
       PARAMETER (MPI_DIST_GRAPH=3)
       INTEGER MPI_VERSION
       PARAMETER (MPI_VERSION=3)
       INTEGER MPI_SUBVERSION
       PARAMETER (MPI_SUBVERSION=1)
       INTEGER MPI_LOCK_EXCLUSIVE
       PARAMETER (MPI_LOCK_EXCLUSIVE=234)
       INTEGER MPI_LOCK_SHARED
       PARAMETER (MPI_LOCK_SHARED=235)
       INTEGER MPI_COMPLEX
       PARAMETER (MPI_COMPLEX=1275070494)
       INTEGER MPI_DOUBLE_COMPLEX
       PARAMETER (MPI_DOUBLE_COMPLEX=1275072546)
       INTEGER MPI_LOGICAL
       PARAMETER (MPI_LOGICAL=1275069469)
       INTEGER MPI_REAL
       PARAMETER (MPI_REAL=1275069468)
       INTEGER MPI_DOUBLE_PRECISION
       PARAMETER (MPI_DOUBLE_PRECISION=1275070495)
       INTEGER MPI_INTEGER
       PARAMETER (MPI_INTEGER=1275069467)
       INTEGER MPI_2INTEGER
       PARAMETER (MPI_2INTEGER=1275070496)
       INTEGER MPI_2DOUBLE_PRECISION
       PARAMETER (MPI_2DOUBLE_PRECISION=1275072547)
       INTEGER MPI_2REAL
       PARAMETER (MPI_2REAL=1275070497)
       INTEGER MPI_CHARACTER
       PARAMETER (MPI_CHARACTER=1275068698)
       INTEGER MPI_BYTE
       PARAMETER (MPI_BYTE=1275068685)
       INTEGER MPI_UB
       PARAMETER (MPI_UB=1275068433)
       INTEGER MPI_LB
       PARAMETER (MPI_LB=1275068432)
       INTEGER MPI_PACKED
       PARAMETER (MPI_PACKED=1275068687)
       INTEGER MPI_INTEGER1
       PARAMETER (MPI_INTEGER1=1275068717)
       INTEGER MPI_INTEGER2
       PARAMETER (MPI_INTEGER2=1275068975)
       INTEGER MPI_INTEGER4
       PARAMETER (MPI_INTEGER4=1275069488)
       INTEGER MPI_INTEGER8
       PARAMETER (MPI_INTEGER8=1275070513)
       INTEGER MPI_INTEGER16
       PARAMETER (MPI_INTEGER16=MPI_DATATYPE_NULL)
       INTEGER MPI_REAL4
       PARAMETER (MPI_REAL4=1275069479)
       INTEGER MPI_REAL8
       PARAMETER (MPI_REAL8=1275070505)
       INTEGER MPI_REAL16
       PARAMETER (MPI_REAL16=1275072555)
       INTEGER MPI_COMPLEX8
       PARAMETER (MPI_COMPLEX8=1275070504)
       INTEGER MPI_COMPLEX16
       PARAMETER (MPI_COMPLEX16=1275072554)
       INTEGER MPI_COMPLEX32
       PARAMETER (MPI_COMPLEX32=1275076652)
       INTEGER MPI_ADDRESS_KIND
       PARAMETER (MPI_ADDRESS_KIND=8)
       INTEGER MPI_OFFSET_KIND
       PARAMETER (MPI_OFFSET_KIND=8)
       INTEGER MPI_COUNT_KIND
       PARAMETER (MPI_COUNT_KIND=8)
       INTEGER MPI_INTEGER_KIND
       PARAMETER (MPI_INTEGER_KIND=4)
       INTEGER MPI_CHAR
       PARAMETER (MPI_CHAR=1275068673)
       INTEGER MPI_SIGNED_CHAR
       PARAMETER (MPI_SIGNED_CHAR=1275068696)
       INTEGER MPI_UNSIGNED_CHAR
       PARAMETER (MPI_UNSIGNED_CHAR=1275068674)
       INTEGER MPI_WCHAR
       PARAMETER (MPI_WCHAR=1275069454)
       INTEGER MPI_SHORT
       PARAMETER (MPI_SHORT=1275068931)
       INTEGER MPI_UNSIGNED_SHORT
       PARAMETER (MPI_UNSIGNED_SHORT=1275068932)
       INTEGER MPI_INT
       PARAMETER (MPI_INT=1275069445)
       INTEGER MPI_UNSIGNED
       PARAMETER (MPI_UNSIGNED=1275069446)
       INTEGER MPI_LONG
       PARAMETER (MPI_LONG=1275070471)
       INTEGER MPI_UNSIGNED_LONG
       PARAMETER (MPI_UNSIGNED_LONG=1275070472)
       INTEGER MPI_FLOAT
       PARAMETER (MPI_FLOAT=1275069450)
       INTEGER MPI_DOUBLE
       PARAMETER (MPI_DOUBLE=1275070475)
       INTEGER MPI_LONG_DOUBLE
       PARAMETER (MPI_LONG_DOUBLE=MPI_DATATYPE_NULL)
       INTEGER MPI_LONG_LONG_INT
       PARAMETER (MPI_LONG_LONG_INT=1275070473)
       INTEGER MPI_UNSIGNED_LONG_LONG
       PARAMETER (MPI_UNSIGNED_LONG_LONG=1275070489)
       INTEGER MPI_LONG_LONG
       PARAMETER (MPI_LONG_LONG=1275070473)
       INTEGER MPI_FLOAT_INT
       PARAMETER (MPI_FLOAT_INT=-1946157056)
       INTEGER MPI_DOUBLE_INT
       PARAMETER (MPI_DOUBLE_INT=-1946157055)
       INTEGER MPI_LONG_INT
       PARAMETER (MPI_LONG_INT=-1946157054)
       INTEGER MPI_SHORT_INT
       PARAMETER (MPI_SHORT_INT=-1946157053)
       INTEGER MPI_2INT
       PARAMETER (MPI_2INT=1275070486)
       INTEGER MPI_LONG_DOUBLE_INT
       PARAMETER (MPI_LONG_DOUBLE_INT=MPI_DATATYPE_NULL)
       INTEGER MPI_INT8_T
       PARAMETER (MPI_INT8_T=1275068727)
       INTEGER MPI_INT16_T
       PARAMETER (MPI_INT16_T=1275068984)
       INTEGER MPI_INT32_T
       PARAMETER (MPI_INT32_T=1275069497)
       INTEGER MPI_INT64_T
       PARAMETER (MPI_INT64_T=1275070522)
       INTEGER MPI_UINT8_T
       PARAMETER (MPI_UINT8_T=1275068731)
       INTEGER MPI_UINT16_T
       PARAMETER (MPI_UINT16_T=1275068988)
       INTEGER MPI_UINT32_T
       PARAMETER (MPI_UINT32_T=1275069501)
       INTEGER MPI_UINT64_T
       PARAMETER (MPI_UINT64_T=1275070526)
       INTEGER MPI_C_BOOL
       PARAMETER (MPI_C_BOOL=1275068735)
       INTEGER MPI_C_FLOAT_COMPLEX
       PARAMETER (MPI_C_FLOAT_COMPLEX=1275070528)
       INTEGER MPI_C_COMPLEX
       PARAMETER (MPI_C_COMPLEX=1275070528)
       INTEGER MPI_C_DOUBLE_COMPLEX
       PARAMETER (MPI_C_DOUBLE_COMPLEX=1275072577)
       INTEGER MPI_C_LONG_DOUBLE_COMPLEX
       PARAMETER (MPI_C_LONG_DOUBLE_COMPLEX=MPI_DATATYPE_NULL)
       INTEGER MPI_AINT
       PARAMETER (MPI_AINT=1275070531)
       INTEGER MPI_OFFSET
       PARAMETER (MPI_OFFSET=1275070532)
       INTEGER MPI_COUNT
       PARAMETER (MPI_COUNT=1275070533)
       INTEGER MPI_CXX_BOOL
       PARAMETER (MPI_CXX_BOOL=1275068723)
       INTEGER MPI_CXX_FLOAT_COMPLEX
       PARAMETER (MPI_CXX_FLOAT_COMPLEX=1275070516)
       INTEGER MPI_CXX_DOUBLE_COMPLEX
       PARAMETER (MPI_CXX_DOUBLE_COMPLEX=1275072565)
       INTEGER MPI_CXX_LONG_DOUBLE_COMPLEX
       PARAMETER (MPI_CXX_LONG_DOUBLE_COMPLEX=201326592)
       INTEGER MPI_COMBINER_NAMED
       PARAMETER (MPI_COMBINER_NAMED=1)
       INTEGER MPI_COMBINER_DUP
       PARAMETER (MPI_COMBINER_DUP=2)
       INTEGER MPI_COMBINER_CONTIGUOUS
       PARAMETER (MPI_COMBINER_CONTIGUOUS=3)
       INTEGER MPI_COMBINER_VECTOR
       PARAMETER (MPI_COMBINER_VECTOR=4)
       INTEGER MPI_COMBINER_HVECTOR_INTEGER
       PARAMETER (MPI_COMBINER_HVECTOR_INTEGER=5)
       INTEGER MPI_COMBINER_HVECTOR
       PARAMETER (MPI_COMBINER_HVECTOR=6)
       INTEGER MPI_COMBINER_INDEXED
       PARAMETER (MPI_COMBINER_INDEXED=7)
       INTEGER MPI_COMBINER_HINDEXED_INTEGER
       PARAMETER (MPI_COMBINER_HINDEXED_INTEGER=8)
       INTEGER MPI_COMBINER_HINDEXED
       PARAMETER (MPI_COMBINER_HINDEXED=9)
       INTEGER MPI_COMBINER_INDEXED_BLOCK
       PARAMETER (MPI_COMBINER_INDEXED_BLOCK=10)
       INTEGER MPI_COMBINER_STRUCT_INTEGER
       PARAMETER (MPI_COMBINER_STRUCT_INTEGER=11)
       INTEGER MPI_COMBINER_STRUCT
       PARAMETER (MPI_COMBINER_STRUCT=12)
       INTEGER MPI_COMBINER_SUBARRAY
       PARAMETER (MPI_COMBINER_SUBARRAY=13)
       INTEGER MPI_COMBINER_DARRAY
       PARAMETER (MPI_COMBINER_DARRAY=14)
       INTEGER MPI_COMBINER_F90_REAL
       PARAMETER (MPI_COMBINER_F90_REAL=15)
       INTEGER MPI_COMBINER_F90_COMPLEX
       PARAMETER (MPI_COMBINER_F90_COMPLEX=16)
       INTEGER MPI_COMBINER_F90_INTEGER
       PARAMETER (MPI_COMBINER_F90_INTEGER=17)
       INTEGER MPI_COMBINER_RESIZED
       PARAMETER (MPI_COMBINER_RESIZED=18)
       INTEGER MPI_COMBINER_HINDEXED_BLOCK
       PARAMETER (MPI_COMBINER_HINDEXED_BLOCK=19)
       INTEGER MPI_TYPECLASS_REAL
       PARAMETER (MPI_TYPECLASS_REAL=1)
       INTEGER MPI_TYPECLASS_INTEGER
       PARAMETER (MPI_TYPECLASS_INTEGER=2)
       INTEGER MPI_TYPECLASS_COMPLEX
       PARAMETER (MPI_TYPECLASS_COMPLEX=3)
       INTEGER MPI_MODE_NOCHECK
       PARAMETER (MPI_MODE_NOCHECK=1024)
       INTEGER MPI_MODE_NOSTORE
       PARAMETER (MPI_MODE_NOSTORE=2048)
       INTEGER MPI_MODE_NOPUT
       PARAMETER (MPI_MODE_NOPUT=4096)
       INTEGER MPI_MODE_NOPRECEDE
       PARAMETER (MPI_MODE_NOPRECEDE=8192)
       INTEGER MPI_MODE_NOSUCCEED
       PARAMETER (MPI_MODE_NOSUCCEED=16384)
       INTEGER MPI_COMM_TYPE_SHARED
       PARAMETER (MPI_COMM_TYPE_SHARED=1)
       INTEGER MPI_MESSAGE_NULL
       PARAMETER (MPI_MESSAGE_NULL=738197504)
       INTEGER MPI_MESSAGE_NO_PROC
       PARAMETER (MPI_MESSAGE_NO_PROC=1811939328)
       INTEGER MPI_THREAD_SINGLE
       PARAMETER (MPI_THREAD_SINGLE=0)
       INTEGER MPI_THREAD_FUNNELED
       PARAMETER (MPI_THREAD_FUNNELED=1)
       INTEGER MPI_THREAD_SERIALIZED
       PARAMETER (MPI_THREAD_SERIALIZED=2)
       INTEGER MPI_THREAD_MULTIPLE
       PARAMETER (MPI_THREAD_MULTIPLE=3)
       INTEGER MPI_MODE_RDONLY
       PARAMETER (MPI_MODE_RDONLY=2)
       INTEGER MPI_MODE_RDWR
       PARAMETER (MPI_MODE_RDWR=8)
       INTEGER MPI_MODE_WRONLY
       PARAMETER (MPI_MODE_WRONLY=4)
       INTEGER MPI_MODE_DELETE_ON_CLOSE
       PARAMETER (MPI_MODE_DELETE_ON_CLOSE=16)
       INTEGER MPI_MODE_UNIQUE_OPEN
       PARAMETER (MPI_MODE_UNIQUE_OPEN=32)
       INTEGER MPI_MODE_CREATE
       PARAMETER (MPI_MODE_CREATE=1)
       INTEGER MPI_MODE_EXCL
       PARAMETER (MPI_MODE_EXCL=64)
       INTEGER MPI_MODE_APPEND
       PARAMETER (MPI_MODE_APPEND=128)
       INTEGER MPI_MODE_SEQUENTIAL
       PARAMETER (MPI_MODE_SEQUENTIAL=256)
       INTEGER MPI_SEEK_SET
       PARAMETER (MPI_SEEK_SET=600)
       INTEGER MPI_SEEK_CUR
       PARAMETER (MPI_SEEK_CUR=602)
       INTEGER MPI_SEEK_END
       PARAMETER (MPI_SEEK_END=604)
       INTEGER MPI_ORDER_C
       PARAMETER (MPI_ORDER_C=56)
       INTEGER MPI_ORDER_FORTRAN
       PARAMETER (MPI_ORDER_FORTRAN=57)
       INTEGER MPI_DISTRIBUTE_BLOCK
       PARAMETER (MPI_DISTRIBUTE_BLOCK=121)
       INTEGER MPI_DISTRIBUTE_CYCLIC
       PARAMETER (MPI_DISTRIBUTE_CYCLIC=122)
       INTEGER MPI_DISTRIBUTE_NONE
       PARAMETER (MPI_DISTRIBUTE_NONE=123)
       INTEGER MPI_DISTRIBUTE_DFLT_DARG
       PARAMETER (MPI_DISTRIBUTE_DFLT_DARG=-49767)
       integer*8 MPI_DISPLACEMENT_CURRENT
       PARAMETER (MPI_DISPLACEMENT_CURRENT=-54278278)
       LOGICAL MPI_SUBARRAYS_SUPPORTED
       PARAMETER(MPI_SUBARRAYS_SUPPORTED=.FALSE.)
       LOGICAL MPI_ASYNC_PROTECTS_NONBLOCKING
       PARAMETER(MPI_ASYNC_PROTECTS_NONBLOCKING=.FALSE.)
       INTEGER MPI_BOTTOM, MPI_IN_PLACE, MPI_UNWEIGHTED
       INTEGER MPI_WEIGHTS_EMPTY
       EXTERNAL MPI_DUP_FN, MPI_NULL_DELETE_FN, MPI_NULL_COPY_FN
       EXTERNAL MPI_WTIME, MPI_WTICK
       EXTERNAL PMPI_WTIME, PMPI_WTICK
       EXTERNAL MPI_COMM_DUP_FN, MPI_COMM_NULL_DELETE_FN
       EXTERNAL MPI_COMM_NULL_COPY_FN
       EXTERNAL MPI_WIN_DUP_FN, MPI_WIN_NULL_DELETE_FN
       EXTERNAL MPI_WIN_NULL_COPY_FN
       EXTERNAL MPI_TYPE_DUP_FN, MPI_TYPE_NULL_DELETE_FN
       EXTERNAL MPI_TYPE_NULL_COPY_FN
       EXTERNAL MPI_CONVERSION_FN_NULL
       REAL*8 MPI_WTIME, MPI_WTICK
       REAL*8 PMPI_WTIME, PMPI_WTICK
      
      
       COMMON /MPIFCMB5/ MPI_UNWEIGHTED
       COMMON /MPIFCMB9/ MPI_WEIGHTS_EMPTY
       SAVE /MPIFCMB5/
       SAVE /MPIFCMB9/
      
       COMMON /MPIPRIV1/ MPI_BOTTOM, MPI_IN_PLACE, MPI_STATUS_IGNORE
      
       COMMON /MPIPRIV2/ MPI_STATUSES_IGNORE, MPI_ERRCODES_IGNORE
       SAVE /MPIPRIV1/,/MPIPRIV2/
      
       COMMON /MPIPRIVC/ MPI_ARGVS_NULL, MPI_ARGV_NULL
       SAVE   /MPIPRIVC/
# 335 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 335 "/home/cmaloney111/Nek5000/core/drive1.f"

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
# 336 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 336 "/home/cmaloney111/Nek5000/core/drive1.f"

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
# 337 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 337 "/home/cmaloney111/Nek5000/core/drive1.f"

# 1 "/home/cmaloney111/Nek5000/core/DPROCMAP" 1
# 1
      logical dProcmapCache
      common /cbpmlo/ dProcmapCache
      
      integer commproc, dProcmapH ! window handle 
      common /cbpmwinh/ commproc, dProcmapH 
      
      integer dProcmapWin 
      common /cbpmwd/ dProcmapWin(2*lelt)
      
      parameter (lur = 80)              ! unsorted size
      parameter (lcu = 8*((lur+8)/8))   ! multiple of 8
      parameter (ls1 = 8*lelt + 2*lcu)  ! larger than unsorted
      parameter (ls2 = lelg   + 2*lcu)  ! larger than unsorted
      parameter (lcs = min(ls1,ls2))    ! not much bigger than lelg
      integer   ucache,cache            ! unsorted and sorted cache
      common /cbpmca/ ucache(lcu,3),cache(lcs,3)
      
      
      
# 338 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 338 "/home/cmaloney111/Nek5000/core/drive1.f"

# 1 "/home/cmaloney111/Nek5000/core/RESTART" 1
c     
c     Restart parameters and variables
c     
# 4
      integer         max_rst
      common /crst_i/ max_rst            ! for full restart
      
      integer nxr,nyr,nzr,nelr,nelgr,istpr,ifiler,nfiler
     $       ,nxo,nyo,nzo,nrg
     $       ,wdsizr,wdsizo
     $       ,nfileo,nproc_o,nfldr
     $       ,er(lelr),nelB,nelBr,npsr
      common /cmfi_i/ nxr,nyr,nzr,nelr,nelgr,istpr,ifiler,nfiler
     $              , nxo,nyo,nzo,nrg
     $              , wdsizr,wdsizo
     $              , nfileo,nproc_o,nfldr
     $              , er,nelB,nelBr,npsr
      
      integer iHeaderSize
      parameter(iHeaderSize=132)
      
      real timer
      common /cmfi_r/ timer
      
      character*3  ihdr
      character*10 rdcode
      character*80 mfi_fname
      common /cmfi_c/ ihdr,rdcode,mfi_fname
      
      character*1  rdcode1(10)
      equivalence (rdcode,rdcode1)
      
      logical ifgetx ,ifgetu ,ifgetp ,ifgett ,ifgtps (ldimt1),ifgtim
     $       ,ifgetxr,ifgetur,ifgetpr,ifgettr,ifgtpsr(ldimt1),ifgtimr
     $       ,if_byte_sw,ifgetz,ifgetw,ifdiro,ifgfldr
      common /cmfi_l/ ifgetx,ifgetu,ifgetp,ifgett,ifgtps,ifgtim
     $       ,ifgetxr,ifgetur,ifgetpr,ifgettr,ifgtpsr,ifgtimr
     $       ,if_byte_sw,ifgetz,ifgetw,ifdiro,ifgfldr
      
      integer         fid0,fid0r,pid0,pid1,pid0r,pid1r,pid00
      common /cmfi_p/ fid0,fid0r,pid0,pid1,pid0r,pid1r,pid00 
      
      integer          ifh_mbyte
      common /i4mpiio/ ifh_mbyte
      
      integer rsH, commrs
      common /cbrewinh/ rsH, commrs
# 339 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 339 "/home/cmaloney111/Nek5000/core/drive1.f"
      
      if(instep.ne.0) call runstat
      
c      if (ifstrs) then
c         call crs_free(xxth_strs) 
c      else
c         call crs_free(xxth(1))
c      endif
      





      

# 355
      if (commrs .ne. MPI_COMM_NULL) then
        call MPI_Win_free(rsH, ierr)
      endif

      
# 360
      call in_situ_end()
      call exitt0()
      
      return
      end
c-----------------------------------------------------------------------
      subroutine nek__multi_advance(kstep,msteps)
      

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
# 369 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 369 "/home/cmaloney111/Nek5000/core/drive1.f"

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
# 370 "/home/cmaloney111/Nek5000/core/drive1.f" 2
# 370 "/home/cmaloney111/Nek5000/core/drive1.f"
      
      do i=1,msteps
         istep = istep+i
         call nek_advance
      
         if (ifneknekc) then 
            call neknek_exchange
            call bcopy
            call chk_outflow
         endif
      enddo
      
      return
      end
c-----------------------------------------------------------------------

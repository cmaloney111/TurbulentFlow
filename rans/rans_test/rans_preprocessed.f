# 1 "rans.f"
      include "experimental/rans_komg.f"
c-----------------------------------------------------------------------c
      subroutine userchk()
# 5 "rans.f"
c      implicit none
# 7 "rans.f"
      include 'SIZE'
      include 'TOTAL'
# 10 "rans.f"
      integer e
# 12 "rans.f"
      common /cdsmag/ ediff(lx1,ly1,lz1,lelv)
# 14 "rans.f"
      real x0(3)
      data x0 /0.0, 0.0, 0.0/
      save x0
# 18 "rans.f"
c     Save the boundary ID(1) with the iobj_wall so it can be used to compute wall shear
      integer bIDs(1)
      save iobj_wall
# 23 "rans.f"
c     This creates the object iobj_wall
      if (istep.eq.0) then
         bIDs(1) = 5
         call create_obj(iobj_wall,bIDs,1)
         nm = iglsum(nmember(iobj_wall),1)
         if(nid.eq.0) write(6,*) 'obj_wall nmem:', nm 
         call prepost(.true.,'   ')
      endif
# 32 "rans.f"
      scale = 2.0 ! CD = F/(.5 rho U^2 ) = 2*F using scale=2 gives us the coefficients      
# 34 "rans.f"
c     Computes drag over wall
      if (istep.eq.0) call set_obj
      call torque_calc(scale,x0,.true.,.false.)
      dragx_avg = dragx(1)
      dragy_avg = dragy(1)
      drag_px = dragpx(1)
      drag_py = dragpy(1)
      drag_vx = dragvx(1)
      drag_vy = dragvy(1)
# 44 "rans.f"
      open(unit=57,file='drag.txt')
# 46 "rans.f"
      if (mod(istep,5000).lt.5) then
      write(57,*) 'Time = ', time,
     +            'Fx_p=', drag_px, 'Fx_v=', drag_vx, 'Fx=', dragx_avg,
     +            'Fy_p=', drag_py, 'Fy_v=', drag_vy, 'Fy=', dragy_avg
      endif
c      nintv = 40000
# 53 "rans.f"
c      if(istep.eq.0.or.(mod(istep,nintv).eq.0)) call write_soln(nintv)
# 55 "rans.f"
      return
      end
c-----------------------------------------------------------------------c
# 61 "rans.f"
c-----------------------------------------------------------------------c
      subroutine set_obj
c
      include 'SIZE'
      include 'TOTAL'
      integer e,f
# 68 "rans.f"
c     Define new objects
# 70 "rans.f"
      nobj = 1                  ! for Periodic
      iobj = 0
      do ii=nhis+1,nhis+nobj
         iobj = iobj+1
         hcode(10,ii) = 'I'
         hcode( 1,ii) = 'F' ! 'F'
         hcode( 2,ii) = 'F' ! 'F'
         hcode( 3,ii) = 'F' ! 'F'
         lochis(1,ii) = iobj
      enddo
      nhis = nhis + nobj
# 82 "rans.f"
      if (maxobj.lt.nobj) write(6,*) 'increase maxobj in SIZEu. rm *.o'
      if (maxobj.lt.nobj) call exitt
# 85 "rans.f"
      nxyz = nx1*ny1*nz1
      do e=1,nelv
      do f=1,2*ndim
         id_face = boundaryID(f,e)
          if (id_face.eq.5) then
            iobj = 1
             if (iobj.gt.0) then
               nmember(iobj) = nmember(iobj) + 1
               mem = nmember(iobj)
               ieg = lglel(e)
               object(iobj,mem,1) = ieg
               object(iobj,mem,2) = f
c              write(6,1) iobj,mem,f,ieg,e,nid,' OBJ'
    1          format(6i9,a4)
            endif
# 101 "rans.f"
         endif
      enddo
      enddo
c     write(6,*) 'number',(nmember(k),k=1,4)
# 106 "rans.f"
      return
      end
# 109 "rans.f"
C----------------------------------------------------------------------
      subroutine uservp (ix,iy,iz,eg)
      implicit none
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
# 116 "rans.f"
      integer e,ix,iy,iz,eg
c      common /rans_usr/ ifld_tke, ifld_tau, m_id
c      integer ifld_tke, ifld_tau, m_id
# 120 "rans.f"
      real rans_mut,rans_mutsk,rans_mutso,rans_turbPrandtl
      real mu_t,Pr_t 
     
      e = gllel(eg)
      
      Pr_t=rans_turbPrandtl()
      mu_t=rans_mut(ix,iy,iz,e)
      
      if(ifield.eq.1) then
        t(ix,iy,iz,e,4)=mu_t/cpfld(ifield,1) !store eddy viscosity for post processing
        udiff  = cpfld(ifield,1)+mu_t
        utrans = cpfld(ifield,2)
      else if(ifield.eq.2) then
        udiff  = cpfld(ifield,1)+mu_t*cpfld(ifield,2)/(Pr_t*cpfld(1,2))
        utrans = cpfld(ifield,2)
      else if(ifield.eq.3) then !use rho and mu from field 1
        udiff  = cpfld(1,1)+rans_mutsk(ix,iy,iz,e)
        utrans = cpfld(1,2)
      else if(ifield.eq.4) then !use rho and mu from field 1
        udiff  = cpfld(1,1)+rans_mutso(ix,iy,iz,e)
        utrans = cpfld(1,2)
      end if
      
      return
      end
      
C-----------------------------------------------------------------------
      subroutine userf  (ix,iy,iz,eg)
# 149 "rans.f"
      include 'SIZE'
      include 'TSTEP'
      include 'NEKUSE'
# 153 "rans.f"
      integer ix,iy,iz,e,eg
# 155 "rans.f"
      ffx = 0.0
      ffy = 0.0
      ffz = 0.0
# 159 "rans.f"
      return
      end
# 162 "rans.f"
C-----------------------------------------------------------------------
      subroutine userq  (ix,iy,iz,eg)
      implicit none
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
# 169 "rans.f"
c      common /rans_usr/ ifld_tke, ifld_tau, m_id
c      integer ifld_tke,ifld_tau, m_id
# 172 "rans.f"
      real rans_kSrc,rans_omgSrc
      real rans_kDiag,rans_omgDiag
# 175 "rans.f"
      integer ie,ix,iy,iz,eg
      ie = gllel(eg)
# 178 "rans.f"
      if (ifield.eq.3) then
        qvol = rans_kSrc  (ix,iy,iz,ie)
        avol = rans_kDiag (ix,iy,iz,ie)
      else if (ifield.eq.4) then
        qvol = rans_omgSrc (ix,iy,iz,ie)
        avol = rans_omgDiag(ix,iy,iz,ie)
      else
        qvol = 0.0
      end if
# 188 "rans.f"
      return
      end
      
C-----------------------------------------------------------------------
      subroutine userbc (ix,iy,iz,iside,eg)
C     NOTE ::: This subroutine MAY NOT be called by every process
      implicit none
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
# 199 "rans.f"
      integer ix,iy,iz,iside,e,eg
      character*3 cb1	
  
      common /rans_usr/ ifld_tke, ifld_tau, m_id
      integer ifld_tke,ifld_tau, m_id
# 205 "rans.f"
      e = gllel(eg)
      cb1 = cbc(iside,e,1) !velocity boundary condition
      
      ux=1.0
      uy=0.0
      uz=0.0
      temp=0.0
      
c      if(cb1.eq.'W  ') then
c        if(ifield.eq.ifld_tke) then
c          temp = 0.0
c        else if(ifield.eq.ifld_tau) then
c          temp = 0.0
c        end if
c      end if
      
      return
      end
      
C-----------------------------------------------------------------------
      subroutine useric (ix,iy,iz,eg) !how does this change for restart?
      implicit none
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
# 231 "rans.f"
      integer ix,iy,iz,e,eg
# 233 "rans.f"
      common /rans_usr/ ifld_tke, ifld_tau, m_id
      integer ifld_tke,ifld_tau, m_id
      
      e = gllel(eg)
# 238 "rans.f"
      ux=1.0 !Maybe this should be 0.0? Or 1.0?
      uy=0.0
      uz=0.0
      temp=0.0
# 243 "rans.f"
      if(ifield.eq.3) temp = 0.01
      if(ifield.eq.4) temp = 0.2
      
      return
      end
      
C-----------------------------------------------------------------------
      subroutine usrdat
# 252 "rans.f"
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
# 256 "rans.f"
c      do i=1,nelt
c      	do j=1,2*ndim
c      	 if(bc(5,j,i,1).eq.4) then
c      	  cbc(j,i,1)='v  '
c         elseif(bc(5,j,i,1).eq.2) then
c       	  cbc(j,i,1)='O  '
c         elseif(bc(5,j,i,1).eq.3) then
c          cbc(j,i,1)='SYM'
c         elseif(bc(5,j,i,1).eq.1) then
c          cbc(j,i,1)='W  '
c         elseif(bc(5,j,i,1).eq.5) then
c          cbc(j,i,1)='W  '
c         endif
c        enddo
c      enddo 
# 272 "rans.f"
      return
      end
C-----------------------------------------------------------------------
      subroutine usrdat2
# 277 "rans.f"
      implicit none
      include 'SIZE'
      include 'TOTAL'
# 281 "rans.f"
      real wd
      common /walldist/ wd(lx1,ly1,lz1,lelv)
# 284 "rans.f"
      common /rans_usr/ ifld_tke, ifld_tau, m_id
      integer ifld_tke,ifld_tau, m_id
      real xmin,xmax,ymin,ymax,scaley,scalex
      real glmin,glmax
      
      integer w_id
      integer n
      real coeffs(30) !array for passing your own coeffs
      logical ifcoeffs
# 296 "rans.f"
      n=nx1*ny1*nz1*nelv
# 299 "rans.f"
C     rescale the domain BEFORE calling rans_init --- probably don't do this!
c      xmin=glmin(xm1,n)
c      xmax=glmax(xm1,n)
c      ymin=glmin(ym1,n)
c      ymax=glmax(ym1,n)
# 305 "rans.f"
c      scalex=3.0/8.0/(xmax-xmin) !make the elements square on average
c      scaley=1.0/(ymax-ymin)
# 308 "rans.f"
c      call cmult(xm1,scalex,n)
c      call cmult(ym1,scaley,n) !unclear if this was necessary, but we will try it
# 311 "rans.f"
      
      ifld_tke = 3 !address of tke equation in t array
      ifld_tau = 4 !address of omega equation in t array
      ifcoeffs =.false. !set to true to pass your own coeffs
# 316 "rans.f"
C     Supported models:
c     m_id = 0 !regularized standard k-omega (no wall functions)
c     m_id = 1 !regularized low-Re k-omega (no wall functions)
c     m_id = 2 !regularized standard k-omega SST (no wall functions)
c     m_id = 3 !Not supported
      m_id = 4 !standard k-tau
c     m_id = 5 !low Re k-tau 
c     m_id = 6 !standard k-tau SST
# 325 "rans.f"
C     Wall distance function:
c     w_id = 0 ! user specified
c     w_id = 1 ! cheap_dist (path to wall, may work better for periodic boundaries)
      w_id = 2 ! distf (coordinate difference, provides smoother function)
# 330 "rans.f"
c      set velocity BCsc BEFORE calling rans_init!!!!
      call setbc(4,1,'v  ') ! inflow
      call setbc(2,1,'O  ') ! outflow
      call setbc(3,1,'SYM') ! bottom
      call setbc(1,1,'SYM') ! top -- unclear if we want SYM or W
      call setbc(5,1,'W  ') ! airfoil
# 340 "rans.f"
      call rans_init(ifld_tke,ifld_tau,ifcoeffs,coeffs,w_id,wd,m_id)
# 342 "rans.f"
      return
      end
      
C-----------------------------------------------------------------------
      subroutine usrdat3
# 348 "rans.f"
      include 'SIZE'
      include 'TOTAL'
# 351 "rans.f"
      
      return
      end
C-----------------------------------------------------------------------
# 357 "rans.f"
c automatically added by makenek
      subroutine usrdat0() 
# 360 "rans.f"
      return
      end
# 363 "rans.f"
c automatically added by makenek
      subroutine usrsetvert(glo_num,nel,nx,ny,nz) ! to modify glo_num
      integer*8 glo_num(1)
# 367 "rans.f"
      return
      end
# 370 "rans.f"
c automatically added by makenek
      subroutine userqtl
# 373 "rans.f"
      call userqtl_scig
# 375 "rans.f"
      return
      end

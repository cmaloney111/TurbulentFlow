      real function test_omgSrc(ix,iy,iz,iel)
      implicit none
      integer ix,iy,iz,iel
      real temp_array(10,10,10,10)
      
      temp_array(1,1,1,1) = 1.0
      test_omgSrc = temp_array(ix,iy,iz,iel)
      
      return
      end

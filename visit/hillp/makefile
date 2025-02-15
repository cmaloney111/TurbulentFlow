### makefile automatically created by makenek 01/28/2025 19:46:46 ###
BINNAME=nek5000
CASENAME:=hillp
CASEDIR:=/home/cmaloney1/Nek5000/run/hillp
S:=/home/cmaloney1/Nek5000
OPT_INCDIR=./ 
OBJDIR=$(CASEDIR)/obj
LIB=$(OBJDIR)/libnek5000.a

LD=mpif77
FC=mpif77
CC=mpicc
FFLAGS+=-fpp -r8 -fpconstant  
CFLAGS+=
LDFLAGS+= -Wl,--allow-multiple-definition

PPPO=
PPS= MPI UNDERSCORE TIMER

USR=
USR_LFLAGS= -L$S/3rd_party/blasLapack -lblasLapack -L$S/3rd_party/gslib/lib -lgs

MPI=1
DPROCMAP=0
VISIT=0

VISIT_IFLAGS=
VISIT_LFLAGS=
HYPRE_IFLAGS=
GSLIB_IFLAGS:= -I$S/3rd_party/gslib/include
PARRSB_IFLAGS=

################################################################################

PPS_F = $(patsubst %,$(PPPO)-D%,$(PPS))
PPS_C = $(patsubst %,-D%,$(PPS))

# CORE##########################################################################
CORE = drive1.o drive2.o \
plan5.o nekio.o \
plan4.o bdry.o coef.o conduct.o connect1.o connect2.o \
dssum.o eigsolv.o gauss.o genxyz.o navier1.o makeq.o \
navier0.o navier2.o navier3.o navier4.o prepost.o speclib.o \
map2.o mvmesh.o ic.o gfldr.o ssolv.o planx.o math.o mxm_wrapper.o \
hmholtz.o subs1.o subs2.o \
gmres.o hsmg.o convect.o convect2.o induct.o perturb.o \
navier5.o navier6.o navier7.o navier8.o fast3d.o fasts.o calcz.o \
byte.o chelpers.o byte_mpi.o postpro.o interp.o \
cvode_driver.o multimesh.o \
vprops.o makeq_aux.o \
papi.o nek_in_situ.o \
reader_rea.o reader_par.o reader_re2.o \
finiparser.o iniparser.o dictionary.o \
hpf.o \
fcrs.o crs_xxt.o crs_amg.o \
fem_amg_preco.o crs_hypre.o \
partitioner.o
################################################################################
# MXM 
MXM=mxm_std.o
################################################################################

COMM_MPI := comm_mpi.o
ifeq ($(MPI),0)
  COMM_MPI := ${COMM_MPI} mpi_dummy.o
endif

$(info $(shell mkdir -p $(OBJDIR)))
DUMMY:= $(shell cp $S/core/PARALLEL.default $(OBJDIR)/PARALLEL 2>/dev/null)
ifeq ($(DPROCMAP),1)
	CORE := ${CORE} dprocmap.o
 	DUMMY:= $(shell cp $S/core/PARALLEL.dprocmap $(OBJDIR)/PARALLEL 2>/dev/null)
endif

ifneq ($(VISIT),0)
  VISITNEK_INCLUDES:=-I$S/core/3rd_party
  VISITO := visit.o 
  USR_LFLAGS := ${USR_LFLAGS} $(VISIT_LFLAGS)
endif

ifeq ($(MPI),0)
	DUMMY:= $(shell cp $S/core/mpi_dummy.h $S/core/mpif.h) 
else
	DUMMY:= $(shell rm -rf $S/core/mpif.h) 
endif

TMP1 = $(CORE) $(MXM) $(USR) $(COMM_MPI) $(VISITO)

INCLUDES_DIR += $(OPT_INCDIR)
INCLUDES_DIR += $(S)/core/experimental
INCLUDES_DIR += $(OBJDIR)
INCLUDES:= $(foreach dir,$(INCLUDES_DIR),-I$(dir))
#$(info $(shell echo "INCLUDES: $(INCLUDES)"))

NOBJS = $(patsubst %,$(OBJDIR)/%,$(TMP1))
USRF = $(OBJDIR)/${CASENAME}.o

L0 = $(G) -O0
L2 = $(G) -O2
L3 = $(G) -O3

FL0   = $(L0) $(FFLAGS) $(PPS_F) -I$(CASEDIR) -I$S/core $(INCLUDES)
FL2   = $(L2) $(FFLAGS) $(PPS_F) -I$(CASEDIR) -I$S/core $(INCLUDES)
FL3   = $(L3) $(FFLAGS) $(PPS_F) -I$(CASEDIR) -I$S/core $(INCLUDES)

cFL0   = $(L0) $(CFLAGS) $(PPS_C) -I$S/core $(INCLUDES)
cFL2   = $(L2) $(CFLAGS) $(PPS_C) -I$S/core $(INCLUDES)
cFL3   = $(L3) $(CFLAGS) $(PPS_C) -I$S/core $(INCLUDES)

################################################################################
all : nek5000

usr: $(USRF)

$(USRF): ./SIZE ${CASENAME}.usr
	@env CASENAME=$(CASENAME) PPS="$(PPS)" $S/core/mkuserfile
	$(FC) $(FL2) $(VISITNEK_INCLUDES) -c $(CASEDIR)/${CASENAME}.f -o $(USRF)

nek5000: ${LIB} $(USRF) 
	$(FC) -c $(FL2) $S/core/drive.f
	$(LD) $(FL2) -o ${BINNAME} drive.o $(USRF) ${LIB} $(USR_LFLAGS) $(LDFLAGS)
	@if test -f ${BINNAME}; then \
        echo ""; \
        echo "$(NEK_WARN)"; \
	echo "#############################################################"; \
	echo "#                  Compilation successful!                  #"; \
	echo "#############################################################"; \
	size ${BINNAME}; \
	echo ""; \
	else \
	echo -e "ERROR: Cannot find ${BINNAME}!"; \
	echo -e ""; \
	fi
	@rm -f $(CASEDIR)/${CASENAME}.f drive.o *.i

lib: ${LIB} 

${LIB}: $(NOBJS)
	@$(AR) cr ${LIB} $(NOBJS)
	@ranlib ${LIB}
	@if test -f ${LIB}; then \
	printf "done\n"; \
	else \
	printf "failed!\n"; \
	echo ""; \
	fi
	@rm -f *.i

clean::
	@rm -rf ${LIB}  ${CASENAME}.f obj ${BINNAME}

.PHONY: all clean lib usr 

$(NOBJS) : ./SIZE

# NEK CORE     ##################################################################
$(OBJDIR)/drive1.o	:$S/core/drive1.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/drive2.o	:$S/core/drive2.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/prepost.o	:$S/core/prepost.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/postpro.o	:$S/core/postpro.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/gfldr.o	:$S/core/gfldr.f $S/core/GFLDR;		$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/connect1.o	:$S/core/connect1.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/connect2.o	:$S/core/connect2.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/genxyz.o	:$S/core/genxyz.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/subs1.o	:$S/core/subs1.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/subs2.o	:$S/core/subs2.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/map2.o	:$S/core/map2.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/dprocmap.o    :$S/core/dprocmap.f $S/core/DPROCMAP;   $(FC) -c $(FL2) $< -o $@
$(OBJDIR)/interp.o      :$S/core/interp.f;                      $(FC) -c $(FL2) $< -o $@
$(OBJDIR)/mvmesh.o	:$S/core/mvmesh.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/induct.o	:$S/core/induct.f;	      		$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/convect.o	:$S/core/convect.f;	      		$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/convect2.o	:$S/core/convect2.f;	      		$(FC) -c $(FL0) $< -o $@
$(OBJDIR)/perturb.o	:$S/core/perturb.f;	      		$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/hsmg.o	:$S/core/hsmg.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/gmres.o	:$S/core/gmres.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/papi.o	:$S/core/papi.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/bdry.o	:$S/core/bdry.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/ic.o		:$S/core/ic.f;			        $(FC) -c $(FL2) $< -o $@
$(OBJDIR)/conduct.o	:$S/core/conduct.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/navier0.o	:$S/core/navier0.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/navier2.o	:$S/core/navier2.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/navier3.o	:$S/core/navier3.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/navier4.o	:$S/core/navier4.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/navier5.o	:$S/core/navier5.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/navier6.o	:$S/core/navier6.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/navier7.o	:$S/core/navier7.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/navier8.o	:$S/core/navier8.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/speclib.o	:$S/core/speclib.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/calcz.o	:$S/core/calcz.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/fast3d.o	:$S/core/fast3d.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/dssum.o	:$S/core/dssum.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/eigsolv.o	:$S/core/eigsolv.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/gauss.o	:$S/core/gauss.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/planx.o	:$S/core/planx.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/ssolv.o	:$S/core/ssolv.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/coef.o	:$S/core/coef.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/plan4.o	:$S/core/plan4.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/plan5.o	:$S/core/plan5.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/vprops.o	:$S/core/vprops.f;	   		$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/cvode_driver.o :$S/core/cvode_driver.f $S/core/CVODE;	$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/makeq.o      	:$S/core/makeq.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/makeq_aux.o  	:$S/core/makeq_aux.f;		        $(FC) -c $(FL2) $< -o $@
$(OBJDIR)/hmholtz.o	:$S/core/hmholtz.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/navier1.o	:$S/core/navier1.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/fasts.o	:$S/core/fasts.f;		        $(FC) -c $(FL2) $< -o $@
$(OBJDIR)/comm_mpi.o	:$S/core/comm_mpi.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/mpi_dummy.o	:$S/core/mpi_dummy.f;		        $(FC) -c $(FL2) $< -o $@
$(OBJDIR)/byte_mpi.o	:$S/core/byte_mpi.f;			$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/math.o	:$S/core/math.f;			$(FC) -c $(FL3) $< -o $@
$(OBJDIR)/multimesh.o	:$S/core/multimesh.f;		        $(FC) -c $(FL2) $< -o $@
$(OBJDIR)/lb_setqvol.o	:$S/core/lb_setqvol.f;		        $(FC) -c $(FL2) $< -o $@
$(OBJDIR)/reader_rea.o	:$S/core/reader_rea.f;	 	        $(FC) -c $(FL2) $< -o $@
$(OBJDIR)/reader_par.o	:$S/core/reader_par.f $S/core/PARDICT;	$(FC) -c $(FL2) $< -o $@
$(OBJDIR)/reader_re2.o	:$S/core/reader_re2.f;		        $(FC) -c $(FL2) $< -o $@
$(OBJDIR)/hpf.o	       	:$S/core/hpf.f;         		$(FC) -c $(FL2) $< -o $@

# MXM       ############################################################################
$(OBJDIR)/mxm_wrapper.o	  :$S/core/mxm_wrapper.f;	$(FC) -c $(FL2) $< -o $@ 
$(OBJDIR)/mxm_std.o	  :$S/core/mxm_std.f;		$(FC) -c $(FL3) $< -o $@
$(OBJDIR)/mxm_bgq.o	  :$S/core/mxm_bgq.f;		$(FC) -c $(FL3) $< -o $@

# C Files ##################################################################################
$(OBJDIR)/byte.o             :$S/core/byte.c;              $(CC) -c $(cFL2) $< -o $@
$(OBJDIR)/chelpers.o         :$S/core/chelpers.c;          $(CC) -c $(cFL2) $< -o $@
$(OBJDIR)/fcrs.o             :$S/core/fcrs.c;              $(CC) -c $(cFL2) $(GSLIB_IFLAGS) $(HYPRE_IFLAGS) $< -o $@
$(OBJDIR)/crs_xxt.o          :$S/core/crs_xxt.c;           $(CC) -c $(cFL2) $(GSLIB_IFLAGS) $< -o $@
$(OBJDIR)/crs_amg.o          :$S/core/crs_amg.c;           $(CC) -c $(cFL2) $(GSLIB_IFLAGS) $< -o $@
$(OBJDIR)/fem_amg_preco.o    :$S/core/experimental/fem_amg_preco.c; $(CC) -c $(cFL2) $(GSLIB_IFLAGS) $(HYPRE_IFLAGS) $< -o $@
$(OBJDIR)/crs_hypre.o        :$S/core/experimental/crs_hypre.c; $(CC) -c $(cFL2) $(GSLIB_IFLAGS) $(HYPRE_IFLAGS) $< -o $@
$(OBJDIR)/partitioner.o      :$S/core/partitioner.c;       $(CC) -c $(cFL2) $(GSLIB_IFLAGS) $(PARRSB_IFLAGS) $< -o $@
$(OBJDIR)/nekio.o          :$S/core/nekio.c;           $(CC) -c $(cFL2) $(GSLIB_IFLAGS) $< -o $@

# 3rd party #######################################################################################
$(OBJDIR)/visit.o        :$S/core/3rd_party/visit.f;	     $(FC) -c $(VISIT_IFLAGS) $(FL2) $< -o $@
$(OBJDIR)/nek_in_situ.o  :$S/core/3rd_party/nek_in_situ.f;   $(FC) -c $(VISIT_IFLAGS) $(FL2) $< -o $@
$(OBJDIR)/finiparser.o   :$S/core/3rd_party/finiparser.c;    $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/iniparser.o    :$S/core/3rd_party/iniparser.c;     $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/dictionary.o   :$S/core/3rd_party/dictionary.c;    $(CC) -c $(cFL2) $(JL) $< -o $@

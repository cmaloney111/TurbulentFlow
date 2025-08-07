### makefile automatically created by makenek 08/01/2025 20:35:07 ###
BINNAME=nek5000
CASENAME:=rans
CASEDIR:=$(shell pwd)
S:=/home/cmaloney111/Nek5000
OPT_INCDIR=./ 
OBJDIR=$(CASEDIR)/obj
LIB=$(OBJDIR)/libnek5000.a

LD=mpif90
FC=mpif90
CC=mpicc
FFLAGS+=-cpp -fdefault-real-8 -fdefault-double-8 -w -std=legacy -fPIC
CFLAGS+=-fPIC
LDFLAGS+=-Wl,--no-relax -Wl,--allow-multiple-definition

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

################################################################################
################################################################################

COMM_MPI := comm_mpi.o
ifeq ($(MPI),0)
  COMM_MPI := ${COMM_MPI} mpi_dummy.o
endif

$(info $(shell mkdir -p $(OBJDIR)))
DUMMY:= $(shell cp $S/core/PARALLEL.default $(OBJDIR)/PARALLEL 2>/dev/null)
ifeq ($(DPROCMAP),1)
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

TMP1 = $(USR)

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
	@rm -rf ${LIB} obj ${BINNAME}

.PHONY: all clean lib usr 

$(NOBJS) : ./SIZE

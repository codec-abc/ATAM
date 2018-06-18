##############################
# ATAM
##############################

include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

set( PROJ_NAME ATAM )
project( ${PROJ_NAME} )

# internal
file(GLOB PROJ_SOURCE *.cpp)
file(GLOB PROJ_SOURCE_C *.c)
file(GLOB PROJ_INCLUDE *.h)

# shared
include_directories(${PROJ_SHARED_CAMERA_DIR})
include_directories(${PROJ_SHARED_TIMER_DIR})
include_directories(${PROJ_SHARED_ATAM_DIR})
include_directories(${PROJ_SHARED_DETECTOR_DIR})
include_directories(${PROJ_SHARED_CALIBRATION_DIR})
include_directories(${PROJ_SHARED_CVSBA_DIR})

if( USE_MULTITHREAD )
	add_definitions( -DMULTITHREAD )
endif()

if( USE_CHESSBOARD )
	add_definitions( -DCHESSBOARD )
endif()

# ExternalProject_Add(
#   CLAPACK
  
#   SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../clapack"
#   #CMAKE_ARGS -DBuildShared=ON -DBuildExamples=OFF -DCMAKE_INSTALL_PREFIX=${GLOBAL_OUTPUT_PATH}/humblelogging
  
#   UPDATE_COMMAND ""
#   INSTALL_COMMAND ""
#   TEST_COMMAND ""
#   PATCH_COMMAND ""
#   #BINARY_DIR "build_clapack"

# )

#add_subdirectory("../clapack")

# target
add_executable(
	${PROJ_NAME}
    ${PROJ_SOURCE}
    ${PROJ_INCLUDE}
    ${PROJ_SOURCE_C}
	${PROJ_SHARED_DATADIR_INCLUDE}
    ${PROJ_SHARED_CAMERA_SOURCE}
    ${PROJ_SHARED_CAMERA_INCLUDE}
    ${PROJ_SHARED_TIMER_SOURCE}
    ${PROJ_SHARED_TIMER_INCLUDE}
    ${PROJ_SHARED_ATAM_SOURCE}
    ${PROJ_SHARED_ATAM_SOURCE_C}
    ${PROJ_SHARED_ATAM_INCLUDE}
	${PROJ_SHARED_DETECTOR_SOURCE}
    ${PROJ_SHARED_DETECTOR_INCLUDE}
	${PROJ_SHARED_CVSBA_SOURCE}
    ${PROJ_SHARED_CVSBA_INCLUDE}
	${PROJ_SHARED_CALIBRATION_SOURCE}
    ${PROJ_SHARED_CALIBRATION_INCLUDE}
)

#add_dependencies(${PROJ_NAME} CLAPACK)

#message(STATUS "${CLAPACK} ")

target_link_libraries(
	${PROJ_NAME}
    ${OpenCV_LIBS}
	${CAM_LIBS}
    ${CLAPACK_LIBS}
    
    # blas.lib
    # lapack.lib
)

#
# Try to find the FreeImage library and include path.
# Once done this will define
#
# FREEIMAGEPLUS_FOUND
# FREEIMAGEPLUS_INCLUDE_PATH
# FREEIMAGEPLUS_LIBRARY
# FREEIMAGEPLUS_LIBRARIES
# 

IF (WIN32)
	FIND_PATH( FREEIMAGEPLUS_INCLUDE_PATH FreeImagePlus.h
		${CMAKE_SOURCE_DIR}/FreeImage
		DOC "The directory where FreeImagePlus.h resides")
	FIND_PATH( FREEIMAGEPLUS_LIB_PATH FreeImagePlus.dll
		"${CMAKE_SOURCE_DIR}/FreeImage/x64"
		DOC "The directory where FreeImagePlus.dll resides")
	FIND_LIBRARY( FREEIMAGEPLUS_LIBRARY
		NAMES FreeImagePlus freeimageplus
		PATHS
		"${CMAKE_SOURCE_DIR}/FreeImage/x64"		
		DOC "The FreeImagePlus library")
ELSE (WIN32)
	FIND_PATH( FREEIMAGEPLUS_INCLUDE_PATH FreeImagePlus.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where FreeImagePlus.h resides")
	FIND_LIBRARY( FREEIMAGEPLUS_LIBRARY
		NAMES FreeImagePlus freeimageplus
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The FreeImagePlus library")
ENDIF (WIN32)

SET(FREEIMAGEPLUS_LIBRARIES ${FREEIMAGEPLUS_LIBRARY})

IF (FREEIMAGEPLUS_INCLUDE_PATH AND FREEIMAGEPLUS_LIBRARY)
	SET( FREEIMAGEPLUS_FOUND TRUE CACHE BOOL "Set to TRUE if FreeImage is found, FALSE otherwise")
ELSE (FREEIMAGEPLUS_INCLUDE_PATH AND FREEIMAGEPLUS_LIBRARY)
	SET( FREEIMAGEPLUS_FOUND FALSE CACHE BOOL "Set to TRUE if FreeImage is found, FALSE otherwise")
ENDIF (FREEIMAGEPLUS_INCLUDE_PATH AND FREEIMAGEPLUS_LIBRARY)

MARK_AS_ADVANCED(
	FREEIMAGEPLUS_FOUND 
	FREEIMAGEPLUS_LIBRARY
	FREEIMAGEPLUS_LIBRARIES
	FREEIMAGEPLUS_INCLUDE_PATH
	FREEIMAGEPLUS_LIB_PATH)

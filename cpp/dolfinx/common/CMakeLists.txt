set(HEADERS_common
  ${CMAKE_CURRENT_SOURCE_DIR}/defines.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_doc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IndexMap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/init.h
  ${CMAKE_CURRENT_SOURCE_DIR}/log.h
  ${CMAKE_CURRENT_SOURCE_DIR}/loguru.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MPI.h
  ${CMAKE_CURRENT_SOURCE_DIR}/subsystem.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Table.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Timer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/TimeLogger.h
  ${CMAKE_CURRENT_SOURCE_DIR}/TimeLogManager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/timing.h
  ${CMAKE_CURRENT_SOURCE_DIR}/types.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UniqueIdGenerator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/version.h
  PARENT_SCOPE)

target_sources(dolfinx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/defines.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IndexMap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/log.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MPI.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/subsystem.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Table.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Timer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TimeLogger.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TimeLogManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/timing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UniqueIdGenerator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

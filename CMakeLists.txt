cmake_minimum_required(VERSION 3.19.3)
add_definitions(-std=c++11)

project(Morpheus)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

find_package(GFlags)
find_package(Glog)
find_package(GLM)
find_package(OpenCV)
find_package(OpenPose)
find_package(Threads)

include_directories(${OpenPose_INCLUDE_DIRS} ${GFLAGS_INCLUDE_DIR} ${GLOG_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
link_directories(~/caffe/build/install/lib)

add_executable(main.bin main.cpp glad.c)

target_link_libraries(main.bin dl)
target_link_libraries(main.bin GL)
target_link_libraries(main.bin glfw)
target_link_libraries(main.bin ${OpenPose_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(main.bin pthread)
target_link_libraries(main.bin Xi)
target_link_libraries(main.bin Xrandr)
target_link_libraries(main.bin X11)

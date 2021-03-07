#define STB_IMAGE_IMPLEMENTATION

#include <iostream>
#include <math.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <openpose/headers.hpp>

#include "shaderprogram.h"
#include "stb_image.h"


const unsigned int DISPLAY_WIDTH = 1920;
const unsigned int DISPLAY_HEIGHT = 1080;
const char* VERT_SHADER_PATH = "/home/wangc21/Desktop/Morpheus/shaders/sample.vert";
const char* FRAG_SHADER_PATH = "/home/wangc21/Desktop/Morpheus/shaders/sample.frag";
const GLfloat LIMB_WIDTH = 50.0f;
const unsigned int CIRCLE_QUALITY = 100;
const GLfloat FACE_RADIUS = 100.0f;

// rectangle limb mappings
int limbMap[14][2] = {  {0, 1},     // neck
                        {2, 1},     // R shoulder
                        {3, 2},     // R bicep
                        {4, 3},     // R forearm
                        {1, 5},     // L shoulder
                        {5, 6},     // L bicep
                        {6, 7},     // L forearm
                        {1, 8},     // torso
                        {9, 8},     // R hip
                        {10, 9},    // R thigh
                        {11, 10},   // R calf
                        {8, 12},    // L hip
                        {12, 13},   // L thigh
                        {13, 14}};  // L calf

// view coords to normalized screen coords
glm::mat4 projection_M = glm::ortho(static_cast<float>(DISPLAY_WIDTH), 0.0f, 
                                    static_cast<float>(DISPLAY_HEIGHT), 0.0f, 
                                    -1.0f, 1.0f);


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

int main(int argc, char* argv[]) {
    // initialize glfw
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // initialize OpenCV
    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        std::cout << "Cannot open camera" << std::endl;
        return -1;
    }
    cam.set(CV_CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH);
    cam.set(CV_CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT);
   
    // initialize OpenPose
    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
    opWrapper.start();

    // create glfw window
    GLFWwindow* window = glfwCreateWindow(DISPLAY_WIDTH, DISPLAY_HEIGHT, "Morpheus", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // load OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // create shader programs
    ShaderProgram sp(VERT_SHADER_PATH, FRAG_SHADER_PATH);
    sp.use();

    // textures
    int texWidth, texHeight, texChannels;
    unsigned char* texData = stbi_load("/home/wangc21/Desktop/Morpheus/textures/white.jpg", &texWidth, &texHeight, &texChannels, 0);
    unsigned int blankTexture;
    glGenTextures(1, &blankTexture);
    glBindTexture(GL_TEXTURE_2D, blankTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, texData);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(texData);

    texData = stbi_load("/home/wangc21/Desktop/Morpheus/textures/skin.jpg", &texWidth, &texHeight, &texChannels, 0);
    unsigned int skinTexture;
    glGenTextures(1, &skinTexture);
    glBindTexture(GL_TEXTURE_2D, skinTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, texData);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(texData);

    // projection transformation, further transformations change model coords
    unsigned int transUni = glGetUniformLocation(sp.ID, "projection");
    glUniformMatrix4fv(transUni, 1, GL_FALSE, glm::value_ptr(projection_M));
    transUni = glGetUniformLocation(sp.ID, "model");
    unsigned int colorUni = glGetUniformLocation(sp.ID, "color");

    // create buffers and buffer data
    unsigned int rectVAO, circVAO, rectVBO, circVBO;
    glGenVertexArrays(1, &rectVAO);
    glGenVertexArrays(1, &circVAO);
    glGenBuffers(1, &rectVBO);
    glGenBuffers(1, &circVBO);

    glBindVertexArray(rectVAO);
    glBindBuffer(GL_ARRAY_BUFFER, rectVBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    // rectangle primitive
    std::vector<GLfloat> rectVerts = {
        0.0f, LIMB_WIDTH,    0.0f, 0.0f,    // bottom left
        1.0f, LIMB_WIDTH,    1.0f, 0.0f,    // bottom right
        0.0f, 0.0f,          0.0f, 1.0f,    // top left
        1.0f, 0.0f,          1.0f, 1.0f     // top right
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * rectVerts.size(), rectVerts.data(), GL_DYNAMIC_DRAW);

    glBindVertexArray(circVAO);
    glBindBuffer(GL_ARRAY_BUFFER, circVBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);
    // circle primitive
    glm::vec2 circleLoc = glm::vec2(0.0f, 0.0f);
    std::vector<GLfloat> circleVerts = {circleLoc.x, circleLoc.y};
    for (int i = 0; i <= CIRCLE_QUALITY; i++) {
        GLfloat theta = i * (360.0f / CIRCLE_QUALITY) * (M_PI / 180);
        GLfloat x = circleLoc.x + cos(theta);
        GLfloat y = circleLoc.y + sin(theta);

        circleVerts.push_back(x);
        circleVerts.push_back(y);
    }
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * circleVerts.size(), circleVerts.data(), GL_DYNAMIC_DRAW);

    // draw in wireframe polygons
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // render loop
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // keyboard input
        processInput(window);

        // read frame
        cv::Mat frame;
        cam >> frame;
        if (frame.empty()) {
            std::cout << "Cannot read frame" << std::endl;
            break;
        }

        // pose inference
        const op::Matrix op_frame = OP_CV2OPCONSTMAT(frame);
        auto data = opWrapper.emplaceAndPop(op_frame);
        if (data != nullptr && !data->empty()) {

            const auto& keypoints = data->at(0)->poseKeypoints;
            
            // if person detected
            if (keypoints.getSize(0) != 0) {

                glm::mat4 model_M;

                // limbs
                glBindVertexArray(rectVAO);
                for (unsigned int i = 0; i < 14; i++) {
                    int idx1 = limbMap[i][0];
                    int idx2 = limbMap[i][1];

                    if (keypoints[{0, idx1, 2}] != 0 && keypoints[{0, idx2, 2}] != 0) {
                        model_M = glm::mat4(1.0f);
                        glm::vec2 coord1 = glm::vec2(keypoints[{0, idx1, 0}], 
                                                     keypoints[{0, idx1, 1}]);
                        glm::vec2 coord2 = glm::vec2(keypoints[{0, idx2, 0}], 
                                                     keypoints[{0, idx2, 1}]);
                        GLfloat length = glm::distance(coord1, coord2);
                        GLfloat theta = atan2(coord2.y - coord1.y, coord2.x - coord1.x);

                        model_M = glm::translate(model_M, glm::vec3(coord1.x + LIMB_WIDTH / 2, 
                                                                    coord1.y + LIMB_WIDTH / 2, 
                                                                    0.0f));
                        model_M = glm::rotate(model_M, theta, glm::vec3(0.0f, 0.0f, 1.0f));
                        model_M = glm::scale(model_M, glm::vec3(length, 1.0f, 1.0f));
                        glUniformMatrix4fv(transUni, 1, GL_FALSE, glm::value_ptr(model_M));
                        //glUniform4f(colorUni, 1.0f, 0.5f, 0.2f, 1.0f);
                        glUniform4f(colorUni, 1.0f, 1.0f, 1.0f, 1.0f);
                        glBindTexture(GL_TEXTURE_2D, skinTexture);
                        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
                    }
                }

                // head
                glBindVertexArray(circVAO);
                if (keypoints[{0, 0, 2}] != 0) {
                    model_M = glm::mat4(1.0f);
                    glm::vec2 noseLoc = glm::vec2(keypoints[{0, 0, 0}], 
                                                  keypoints[{0, 0, 1}]);

                    // scale face by nose-ear distance
                    GLfloat faceRadius = FACE_RADIUS;
                    if (keypoints[{0, 17, 2}] != 0) {
                        // right ear detected
                        glm::vec2 earLoc = glm::vec2(keypoints[{0, 17, 0}], 
                                                     keypoints[{0, 17, 1}]);
                        faceRadius = glm::distance(noseLoc, earLoc);
                    } else if (keypoints[{0, 18, 2}] != 0) {
                        // left ear detected
                        glm::vec2 earLoc = glm::vec2(keypoints[{0, 18, 0}], 
                                                     keypoints[{0, 18, 1}]);
                        faceRadius = glm::distance(noseLoc, earLoc);
                    }

                    model_M = glm::translate(model_M, glm::vec3(noseLoc.x - circleLoc.x, 
                                                                noseLoc.y - circleLoc.y, 
                                                                0.0f));
                    model_M = glm::scale(model_M, glm::vec3(faceRadius, faceRadius, faceRadius));
                    glUniformMatrix4fv(transUni, 1, GL_FALSE, glm::value_ptr(model_M));
                    glUniform4f(colorUni, 1.0f, 0.5f, 0.2f, 1.0f);
                    glBindTexture(GL_TEXTURE_2D, blankTexture);
                    glDrawArrays(GL_TRIANGLE_FAN, 0, CIRCLE_QUALITY + 2);

                    // eyes
                    for (int i = 15; i <= 16; i++) {
                        if (keypoints[{0, i, 2}] != 0) {
                            model_M = glm::mat4(1.0f);
                            glm::vec2 eyeLoc = glm::vec2(keypoints[{0, i, 0}], 
                                                         keypoints[{0, i, 1}]);

                            model_M = glm::translate(model_M, glm::vec3(eyeLoc.x - circleLoc.x, 
                                                                        eyeLoc.y - circleLoc.y, 
                                                                        0.0f));
                            model_M = glm::scale(model_M, glm::vec3(faceRadius / 5, faceRadius / 5, faceRadius / 5));
                            glUniformMatrix4fv(transUni, 1, GL_FALSE, glm::value_ptr(model_M));
                            glUniform4f(colorUni, 0.0f, 0.0f, 0.0f, 1.0f);
                            glBindTexture(GL_TEXTURE_2D, blankTexture);
                            glDrawArrays(GL_TRIANGLE_FAN, 0, CIRCLE_QUALITY + 2);
                        }
                    }
                }
            }
        } else {
            std::cout << "Null or empty processed data" << std::endl;
        }

        // swap buffers, poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // de-allocate resources
    cam.release();
    glDeleteVertexArrays(1, &rectVAO);
    glDeleteVertexArrays(1, &circVAO);
    glDeleteBuffers(1, &rectVBO);
    glDeleteBuffers(1, &circVBO);
    sp.free();
    glfwTerminate();
    return 0;
}

// process keyboard input
void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: window resize callback function
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // set viewport to window dimensions
    glViewport(0, 0, width, height);
}

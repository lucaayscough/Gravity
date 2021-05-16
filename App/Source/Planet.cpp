#include "Planet.h"


// Main Planet class.

Planet::Planet()
    : m_diameter(50){}

Planet::~Planet(){}

void Planet::paint(Graphics& g){
    g.setColour(juce::Colours::red);
    g.fillEllipse(0, 0, m_diameter, m_diameter);
}

void Planet::resized(){
    setSize(m_diameter, m_diameter);
}

void Planet::setDiameter(int diameter){
    m_diameter = diameter;
}

void Planet::setEdges(int width, int height){
    m_window_width = width;
    m_window_height = height;
}

void Planet::mouseDown(const MouseEvent& e){
    m_dragger.startDraggingComponent(this, e);
}

void Planet::mouseDrag(const MouseEvent& e){
    m_dragger.dragComponent(this, e, nullptr);
    checkBounds();
}

void Planet::checkBounds(){
    auto posX = getX();
    auto posY = getY();

    if(posX < 0)
        setBounds(0, posY, m_diameter, m_diameter);

    posX = getX();
    posY = getY();

    if(posY < 0)
        setBounds(posX, 0, m_diameter, m_diameter);
    
    posX = getX();
    posY = getY();

    if(posX + m_diameter > m_window_width)
        setBounds(m_window_width - m_diameter, posY, m_diameter, m_diameter);

    posX = getX();
    posY = getY();

    if(posY + m_diameter > m_window_height)
        setBounds(posX, m_window_height - m_diameter, m_diameter, m_diameter);
    
    Logger::writeToLog("X: " + std::to_string(posX) + ", Y: " + std::to_string(posY));
}


// Sun class which inherits from the parent Planet class.
// Movement is disabled and position is fixed.

Sun::Sun(){}

Sun::~Sun(){}

void Sun::paint(Graphics& g){
    setSize(M_DIAMETER, M_DIAMETER);
    g.setColour(juce::Colours::yellow);
    g.fillEllipse(0, 0, M_DIAMETER, M_DIAMETER);
}

void Sun::resized(){}

int Sun::getDiameter(){
    return M_DIAMETER;
}

void Sun::mouseDown(const MouseEvent& e){
    Logger::writeToLog("Generating sample...");
    getNewSample();
}

void Sun::mouseDrag(const MouseEvent& e){}
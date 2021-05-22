#include "Planet.h"


// Main Planet class.

Planet::Planet()
    : m_Diameter(50){}

Planet::Planet(const Planet&){}

Planet::~Planet(){}

void Planet::paint(Graphics& g){
    g.setColour(juce::Colours::red);
    g.fillEllipse(0, 0, m_Diameter, m_Diameter);
}

void Planet::resized(){
    setSize(m_Diameter, m_Diameter);
}

void Planet::setDiameter(int diameter){
    m_Diameter = diameter;
}

void Planet::setWindowBoundary(int width, int height){
    m_WindowWidth = width;
    m_WindowHeight = height;
}

void Planet::mouseDown(const MouseEvent& e){
    m_Dragger.startDraggingComponent(this, e);
}

void Planet::mouseDrag(const MouseEvent& e){
    m_Dragger.dragComponent(this, e, nullptr);
    checkBounds();
}

void Planet::checkBounds(){
    auto posX = getX();
    auto posY = getY();

    if(posX < 0)
        setBounds(0, posY, m_Diameter, m_Diameter);

    posX = getX();
    posY = getY();

    if(posY < 0)
        setBounds(posX, 0, m_Diameter, m_Diameter);
    
    posX = getX();
    posY = getY();

    if(posX + m_Diameter > m_WindowWidth)
        setBounds(m_WindowWidth - m_Diameter, posY, m_Diameter, m_Diameter);

    posX = getX();
    posY = getY();

    if(posY + m_Diameter > m_WindowHeight)
        setBounds(posX, m_WindowHeight - m_Diameter, m_Diameter, m_Diameter);
    
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
#include "Planet.h"


// Main Planet class.

// Constructors and destructors.

Planet::Planet(){}

Planet::Planet(const Planet&){}

Planet::~Planet(){}


// Public methods.

void Planet::paint(Graphics& g){
    g.setColour(juce::Colours::red);
    g.fillEllipse(0, 0, m_Diameter, m_Diameter);
}

void Planet::resized(){
    setSize(m_Diameter, m_Diameter);
}

void Planet::reDraw(int diameter, int x, int y){
    // When called the component is redrawn.
    // If the diameter is modified the difference is split
    // so that the planet remains centred.

    int new_x = x;
    int new_y = y;

    if(diameter > getDiameter()){
        new_x = x - (M_SIZE_MODIFIER / 2);
        new_y = y - (M_SIZE_MODIFIER / 2);
    } else if(diameter < getDiameter()){
        new_x = x + (M_SIZE_MODIFIER / 2);
        new_y = y + (M_SIZE_MODIFIER / 2);
    }

    setDiameter(diameter);
    setBounds(new_x, new_y, diameter, diameter);
}

void Planet::setDiameter(int diameter){
    m_Diameter = diameter;
}

void Planet::setMapBoundaries(int width, int height){
    m_WindowWidth = width;
    m_WindowHeight = height;
}

int Planet::getDiameter(){
    return m_Diameter;
}


// Private methods.

void Planet::mouseDown(const MouseEvent& e){
    m_Dragger.startDraggingComponent(this, e);
}

void Planet::mouseDrag(const MouseEvent& e){
    m_Dragger.dragComponent(this, e, nullptr);
    checkBounds();
}

void Planet::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){
    // If the mouse wheel is moved the diameter of the planet is
    // modified making sure it is not going over the size limitations.
    
    Logger::writeToLog("Wheel moved.");

    if(w.deltaY > 0.0f && getDiameter() < M_MAX_PLANET_SIZE)
        reDraw(getDiameter() + M_SIZE_MODIFIER, getX(), getY());
    else if(w.deltaY < 0.0f && getDiameter() > M_MIN_PLANET_SIZE)
        reDraw(getDiameter() - M_SIZE_MODIFIER, getX(), getY());
}

void Planet::visibilityChanged(){
    Logger::writeToLog("Visibility changed.");
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
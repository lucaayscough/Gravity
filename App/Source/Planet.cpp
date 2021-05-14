#include "Planet.h"


// Main Planet class.

void Planet::paint(Graphics& g){
    g.setColour(juce::Colours::red);
    g.fillEllipse(0, 0, diameter, diameter);
}

void Planet::resized(){
    setSize(diameter, diameter);
}

void Planet::mouseDown(const MouseEvent& e){
    dragger.startDraggingComponent(this, e);
}

void Planet::mouseDrag(const MouseEvent& e){
    dragger.dragComponent(this, e, nullptr);
    checkBounds();
}

void Planet::checkBounds(){
    auto posX = getX();
    auto posY = getY();

    if(posX < 0)
        setBounds(0, posY, diameter, diameter);

    posX = getX();
    posY = getY();

    if(posY < 0)
        setBounds(posX, 0, diameter, diameter);
    
    Logger::writeToLog("X: " + std::to_string(posX) + ", Y: " + std::to_string(posY));
}


// Sun class which inherits from the parent Planet class.
// Movement is disabled and position is fixed.

void Sun::paint(Graphics& g){
    setSize(DIAMETER, DIAMETER);
    g.setColour(juce::Colours::yellow);
    g.fillEllipse(0, 0, DIAMETER, DIAMETER);
}

void Sun::resized(){}
void Sun::mouseDown(const MouseEvent& e){
    Logger::writeToLog("Generating sample...");
    getNewSample();
}
void Sun::mouseDrag(const MouseEvent& e){}
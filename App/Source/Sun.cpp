#include "Sun.h"


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
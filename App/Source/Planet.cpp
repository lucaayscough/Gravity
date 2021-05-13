#include "Planet.h"


void Planet::paint(Graphics& g){
    g.fillAll(juce::Colours::red);
}

void Planet::mouseDown(const MouseEvent& e){
    dragger.startDraggingComponent(this, e);
}
void Planet::mouseDrag(const MouseEvent& e){
    dragger.dragComponent(this, e, nullptr);
}
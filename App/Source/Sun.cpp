#include "Headers.h"


// Sun class which inherits from the parent Planet class.
// Movement is disabled and position is fixed.

Sun::Sun(){}

Sun::Sun(Generator* generator_ptr)
    : Planet(generator_ptr){}

Sun::~Sun(){}

void Sun::paint(Graphics& g){
    setSize(M_DIAMETER, M_DIAMETER);
    g.fillAll(juce::Colours::green);
    g.setColour(juce::Colours::yellow);
    g.fillEllipse(0, 0, M_DIAMETER, M_DIAMETER);
}

void Sun::resized(){}

void Sun::draw(){
    // When called the component is redrawn.
    // setBounds(x, y, diameter, diameter)
    setSize(M_DIAMETER, M_DIAMETER);
    setCentreRelative(0.5, 0.5);
}

int Sun::getDiameter(){
    return M_DIAMETER;
}

void Sun::generateLatents(){
    m_Latents = m_GeneratorPtr->generateLatents();
}

void Sun::generateSample(){
    m_Sample = m_GeneratorPtr->generateSample(m_Latents);
}

bool Sun::hitTest(int x, int y){
    float a = pow((float)x - ((float)M_DIAMETER / 2.0f), 2.0f);
    float b = pow((float)y - ((float)M_DIAMETER / 2.0f), 2.0f);
    float c = sqrt(a + b);

    return c <= (float)(M_DIAMETER / 2);
}

void Sun::mouseDown(const MouseEvent& e){
    if(e.getNumberOfClicks() > 1 && e.mods.isLeftButtonDown()){
        Logger::writeToLog("Generating sample...");

        generateLatents();
        generateSample();

        Logger::writeToLog("Sample generated.");
    }

    else if(e.mods.isLeftButtonDown() && e.mouseWasClicked()){
        Logger::writeToLog("Playing audio...");
        AudioContainer::audio.clear();
        AudioContainer::audio.addArray(m_Sample);
        AudioContainer::playAudio = true;
    }
}

void Sun::mouseDrag(const MouseEvent& e){}

void Sun::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){}
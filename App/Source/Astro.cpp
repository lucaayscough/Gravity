#include "Headers.h"


//--------------------------------------------------//
// Constructors and destructors.

Astro::Astro(juce::String& id, AudioContainer& audiocontainer, Parameters& parameters, ControlPanel& controlpanel)
    :   m_AudioContainerRef(audiocontainer), m_ParametersRef(parameters), m_ControlPanelRef(controlpanel){
    setComponentID(id);
}

Astro::~Astro(){}

//--------------------------------------------------//
// Interface methods.

void Astro::draw(){setBounds(getPosX(), getPosY(), getDiameterWithClipBoundary(), getDiameterWithClipBoundary());}
void Astro::draw(const int diameter, const int x, const int y){setBounds(x, y, diameter + getClipBoundary(), diameter + getClipBoundary());}

//--------------------------------------------------//
// Interface methods.

void Astro::setArea(const float area){getState().setProperty(Parameters::areaProp, area, nullptr);}

void Astro::setPosXY(const int x, const int y){
    getState().setProperty(Parameters::posXProp, x, nullptr);
    getState().setProperty(Parameters::posYProp, y, nullptr);
    setCentrePosXY(x + getRadius(), y + getRadius());
}

void Astro::setCentrePosXY(const int x, const int y){
    getState().setProperty(Parameters::posCentreXProp, x + getClipBoundary() / 2, nullptr);
    getState().setProperty(Parameters::posCentreYProp, y + getClipBoundary() / 2, nullptr);
}

float Astro::getArea(){return getState().getProperty(Parameters::areaProp);}

int Astro::getDiameter(){
    return (int)(round((sqrt(getArea() / 3.1415f) * 2.0f) / 2.0f) * 2.0f);
}

int Astro::getDiameterWithClipBoundary(){return getDiameter() + getClipBoundary();}
int Astro::getRadius(){return getDiameter() / 2;}
int Astro::getRadiusWithClipBoundary(){return (getDiameter() + getClipBoundary()) / 2;}
int Astro::getPosX(){return getState().getProperty(Parameters::posXProp);}
int Astro::getPosY(){return getState().getProperty(Parameters::posYProp);}
int Astro::getCentreX(){return getState().getProperty(Parameters::posCentreXProp);}
int Astro::getCentreY(){return getState().getProperty(Parameters::posCentreYProp);}

float Astro::getDistance(const int xa, const int ya, const int xb, const int yb){
    // TODO:
    // This needs to be moved to ValueTree.

    float a = (float)pow(xb - xa, 2);
    float b = (float)pow(yb - ya, 2); 
    return sqrt(a + b);
}

float Astro::getDistance(Astro* astro_a, Astro* astro_b){
    // TODO:
    // This needs to be moved to ValueTree.

    int centreXA = astro_a->getCentreX();
    int centreYA = astro_a->getCentreY();
    int centreXB = astro_b->getCentreX();
    int centreYB = astro_b->getCentreY();

    float a = (float)pow(centreXB - centreXA, 2);
    float b = (float)pow(centreYB - centreYA, 2);

    return sqrt(a + b);
}

int Astro::getClipBoundary(){return Variables::CLIP_BOUNDARY;}

void Astro::updateGraph(){getState().setProperty(Parameters::updateGraphSignal, true, nullptr);}
void Astro::generateSample(){getState().setProperty(Parameters::generateSampleSignal, true, nullptr);}

void Astro::playSample(){
    m_ParametersRef.setActivePlanet(getState());
    m_AudioContainerRef.sampleIndex.clear();
    m_AudioContainerRef.playAudio = true;
}

//--------------------------------------------------//
// Controller methods.

bool Astro::hitTest(const int x, const int y){
    float a = pow((float)x - (float)getRadiusWithClipBoundary(), 2.0f);
    float b = pow((float)y - (float)getRadiusWithClipBoundary(), 2.0f);
    return sqrt(a + b) <= getRadius();
}

void Astro::mouseEnter(const MouseEvent& e){
    juce::ignoreUnused(e);

    m_ShowForceVectors.setValue(true);
    m_ControlPanelRef.show(getState());
}

void Astro::mouseExit(const MouseEvent& e){
    juce::ignoreUnused(e);

    m_ShowForceVectors.setValue(false);
    m_ControlPanelRef.unshow();
}

void Astro::valueChanged(juce::Value& value){
    juce::ignoreUnused(value);
    repaint();
}

//--------------------------------------------------//
// Animator class.

//--------------------------------------------------//
// Constructors and destructors.

Astro::Animator::Animator(){
    m_DiameterShift.setValue(0);
    startTimer(17);
}

Astro::Animator::~Animator(){
    stopTimer();
}

//--------------------------------------------------//
// Interface methods.

int Astro::Animator::getDiameterShift(){return (int)m_DiameterShift.getValue();}

//--------------------------------------------------//
// Callback methods.

void Astro::Animator::timerCallback(){
    if((int)m_DiameterShift.getValue() >= 20){
        m_DiameterShiftDirection = false;
    }
    else if((int)m_DiameterShift.getValue() <= -20){
        m_DiameterShiftDirection = true;
    }

    if(m_DiameterShiftDirection == true){
        m_DiameterShift.setValue((int)m_DiameterShift.getValue() + 1);
    }
    else{
        m_DiameterShift.setValue((int)m_DiameterShift.getValue() - 1);
    }
}

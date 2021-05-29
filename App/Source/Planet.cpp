#include "Headers.h"


// Main Planet class.

//--------------------------------------------------//
// Constructors and destructors.

Planet::Planet(){}

Planet::Planet(Generator* generator_ptr)
    : m_GeneratorPtr(generator_ptr)
    {
        allocateStorage();

        // Generate random sample.
        generateLatents();
        generateSample();

        // Listener value used to determine when to destroy the planet.
        m_Destroy.setValue(false);
    }

Planet::~Planet(){}


//--------------------------------------------------//
// Public methods.

void Planet::paint(Graphics& g){
    g.setColour(juce::Colours::red);
    draw(m_Diameter, getX(), getY());
    g.fillEllipse(m_ClipBoundary / 2, m_ClipBoundary / 2, m_Diameter, m_Diameter);
}

void Planet::resized(){}

void Planet::draw(int diameter, int x, int y){
    // When called the component is redrawn.
    setBounds(x, y, diameter + m_ClipBoundary, diameter + m_ClipBoundary);
}

void Planet::resizePlanet(int diameter){
    // When called the planet will be resized.
    // X and Y are calculated such that the planet will remain centred.

    int new_x;
    int new_y;

    if(diameter > getDiameter()){
        new_x = getX() - (M_SIZE_MODIFIER / 2);
        new_y = getY() - (M_SIZE_MODIFIER / 2);
    } else{
        new_x = getX() + (M_SIZE_MODIFIER / 2);
        new_y = getY() + (M_SIZE_MODIFIER / 2);
    }

    setDiameter(diameter);
    draw(diameter, new_x, new_y);
}

void Planet::setDiameter(int diameter){
    m_Diameter = diameter;
}

void Planet::setMapBoundaries(int width, int height){
    // Sets the map boundaries.
    m_MapWidth = width;
    m_MapHeight = height;
}

int Planet::getDiameter(){
    return m_Diameter;
}

int Planet::getClipBoundary(){
    return m_ClipBoundary;
}

void Planet::generateLatents(){
    m_Latents = m_GeneratorPtr->generateLatents();
}

void Planet::generateSample(){
    m_Sample = m_GeneratorPtr->generateSample(m_Latents);
}


//--------------------------------------------------//
// Private methods.

bool Planet::hitTest(int x, int y){
    float a = pow((float)x - ((float)m_Diameter + (float)m_ClipBoundary) / 2.0f, 2.0f);
    float b = pow((float)y - ((float)m_Diameter + (float)m_ClipBoundary) / 2.0f, 2.0f);
    float c = sqrt(a + b);

    return c <= m_Diameter / 2;
}

void Planet::mouseDown(const MouseEvent& e){
    // Starts dragging component.
    m_Dragger.startDraggingComponent(this, e);
    
    if(e.mods.isLeftButtonDown()){

        // Generates new sample if double clicked with left mouse button.
        if(e.getNumberOfClicks() > 1){
            Logger::writeToLog("Generating sample...");

            generateLatents();
            generateSample();

            Logger::writeToLog("Sample generated.");
        }
        
        // Plays sample if clicked once with left mouse button.
        else if(e.mouseWasClicked()){
            Logger::writeToLog("Playing audio...");
            AudioContainer::audio.clear();
            AudioContainer::audio.addArray(m_Sample);
            AudioContainer::playAudio = true;
        }
    }
    
    // Destroys planet if clicked with right mouse button.
    else if(e.mods.isRightButtonDown()){
        // Initializes planet destruction.
        m_Destroy.setValue(true);
        Logger::writeToLog("Set to destroy.");
    }    
}

void Planet::mouseUp(const MouseEvent& e){}

void Planet::mouseDrag(const MouseEvent& e){
    m_Dragger.dragComponent(this, e, nullptr);
    checkBounds();
}

void Planet::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){
    // If the mouse wheel is moved the diameter of the planet is
    // modified making sure it is not going over the size limitations.
    
    Logger::writeToLog("Wheel moved.");

    if(w.deltaY > 0.0f && m_Diameter < M_MAX_PLANET_SIZE){
        resizePlanet(m_Diameter + M_SIZE_MODIFIER);
    }
    else if(w.deltaY < 0.0f && m_Diameter > M_MIN_PLANET_SIZE){
        resizePlanet(m_Diameter - M_SIZE_MODIFIER);
    }
}

void Planet::visibilityChanged(){
    Logger::writeToLog("Visibility changed.");
}

void Planet::checkBounds(){
    //Check left boundary.
    if(getX() < -(m_ClipBoundary / 2))
        draw(m_Diameter, -(m_ClipBoundary / 2), getY());

    // Check top boundary.
    if(getY() < -(m_ClipBoundary / 2))
        draw(m_Diameter, getX(), -(m_ClipBoundary / 2));

    // Check right boundary,
    if(getX() + m_Diameter + (m_ClipBoundary / 2) > m_MapWidth)
        draw(m_Diameter, m_MapWidth - m_Diameter - (m_ClipBoundary / 2), getY());

    // Check bottom boundary.
    if(getY() + m_Diameter + (m_ClipBoundary / 2) > m_MapHeight)
        draw(m_Diameter, getX(), m_MapHeight - m_Diameter - (m_ClipBoundary / 2));
    
    // Write planet position to screen.
    Logger::writeToLog("X: " + std::to_string(getX()) + ", Y: " + std::to_string(getY()));
}

void Planet::allocateStorage(){
    // Allocates storage to array that holds sample.
    m_Sample.ensureStorageAllocated(m_GeneratorPtr->M_NUM_SAMPLES);
}

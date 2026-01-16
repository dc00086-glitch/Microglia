import { createContext, useContext, useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';

const HoneymoonContext = createContext();

const defaultItinerary = [
  {
    id: uuidv4(),
    day: 1,
    date: '2026-03-01',
    city: 'Paris',
    country: 'France',
    activities: [
      { id: uuidv4(), time: '09:00', title: 'Arrive at Charles de Gaulle Airport', notes: 'Flight lands at 9am' },
      { id: uuidv4(), time: '12:00', title: 'Check into Hotel', notes: 'Le Marais district' },
      { id: uuidv4(), time: '15:00', title: 'Walk along the Seine', notes: 'Take photos!' },
      { id: uuidv4(), time: '19:00', title: 'Romantic dinner', notes: 'Reserve at Le Jules Verne' },
    ]
  },
  {
    id: uuidv4(),
    day: 2,
    date: '2026-03-02',
    city: 'Paris',
    country: 'France',
    activities: [
      { id: uuidv4(), time: '10:00', title: 'Eiffel Tower visit', notes: 'Book skip-the-line tickets' },
      { id: uuidv4(), time: '14:00', title: 'Louvre Museum', notes: 'See Mona Lisa' },
      { id: uuidv4(), time: '19:00', title: 'Montmartre evening', notes: 'Sacré-Cœur sunset' },
    ]
  },
  {
    id: uuidv4(),
    day: 3,
    date: '2026-03-03',
    city: 'Paris',
    country: 'France',
    activities: [
      { id: uuidv4(), time: '09:00', title: 'Versailles Day Trip', notes: 'Full day excursion' },
    ]
  },
];

const defaultBookings = [
  {
    id: uuidv4(),
    type: 'flight',
    title: 'Flight to Paris',
    confirmationNumber: 'ABC123',
    date: '2026-03-01',
    time: '06:00',
    details: 'Non-stop from JFK to CDG',
    cost: 1200,
  },
  {
    id: uuidv4(),
    type: 'hotel',
    title: 'Hotel Le Marais',
    confirmationNumber: 'HLM456',
    date: '2026-03-01',
    checkOut: '2026-03-05',
    details: 'Deluxe suite with Eiffel view',
    cost: 1800,
  },
  {
    id: uuidv4(),
    type: 'activity',
    title: 'Eiffel Tower Skip-the-Line',
    confirmationNumber: 'EIF789',
    date: '2026-03-02',
    time: '10:00',
    details: 'Summit access included',
    cost: 80,
  },
];

const defaultScrapbook = [
  {
    id: uuidv4(),
    date: '2026-03-01',
    city: 'Paris',
    title: 'Our First Day in Paris',
    description: 'We finally arrived! The city of love welcomed us with open arms.',
    mood: 'romantic',
    photos: [],
  },
];

export function HoneymoonProvider({ children }) {
  const [tripInfo, setTripInfo] = useState(() => {
    const saved = localStorage.getItem('honeymoon-trip-info');
    return saved ? JSON.parse(saved) : {
      couple: 'The Happy Couple',
      startDate: '2026-03-01',
      endDate: '2026-03-24',
      destinations: ['Paris', 'Rome', 'Barcelona', 'Santorini'],
    };
  });

  const [itinerary, setItinerary] = useState(() => {
    const saved = localStorage.getItem('honeymoon-itinerary');
    return saved ? JSON.parse(saved) : defaultItinerary;
  });

  const [bookings, setBookings] = useState(() => {
    const saved = localStorage.getItem('honeymoon-bookings');
    return saved ? JSON.parse(saved) : defaultBookings;
  });

  const [scrapbook, setScrapbook] = useState(() => {
    const saved = localStorage.getItem('honeymoon-scrapbook');
    return saved ? JSON.parse(saved) : defaultScrapbook;
  });

  useEffect(() => {
    localStorage.setItem('honeymoon-trip-info', JSON.stringify(tripInfo));
  }, [tripInfo]);

  useEffect(() => {
    localStorage.setItem('honeymoon-itinerary', JSON.stringify(itinerary));
  }, [itinerary]);

  useEffect(() => {
    localStorage.setItem('honeymoon-bookings', JSON.stringify(bookings));
  }, [bookings]);

  useEffect(() => {
    localStorage.setItem('honeymoon-scrapbook', JSON.stringify(scrapbook));
  }, [scrapbook]);

  // Itinerary functions
  const addDay = (dayData) => {
    setItinerary([...itinerary, { ...dayData, id: uuidv4() }]);
  };

  const updateDay = (dayId, dayData) => {
    setItinerary(itinerary.map(day =>
      day.id === dayId ? { ...day, ...dayData } : day
    ));
  };

  const deleteDay = (dayId) => {
    setItinerary(itinerary.filter(day => day.id !== dayId));
  };

  const addActivity = (dayId, activity) => {
    setItinerary(itinerary.map(day =>
      day.id === dayId
        ? { ...day, activities: [...day.activities, { ...activity, id: uuidv4() }] }
        : day
    ));
  };

  const updateActivity = (dayId, activityId, activityData) => {
    setItinerary(itinerary.map(day =>
      day.id === dayId
        ? {
            ...day,
            activities: day.activities.map(act =>
              act.id === activityId ? { ...act, ...activityData } : act
            )
          }
        : day
    ));
  };

  const deleteActivity = (dayId, activityId) => {
    setItinerary(itinerary.map(day =>
      day.id === dayId
        ? { ...day, activities: day.activities.filter(act => act.id !== activityId) }
        : day
    ));
  };

  // Booking functions
  const addBooking = (booking) => {
    setBookings([...bookings, { ...booking, id: uuidv4() }]);
  };

  const updateBooking = (bookingId, bookingData) => {
    setBookings(bookings.map(booking =>
      booking.id === bookingId ? { ...booking, ...bookingData } : booking
    ));
  };

  const deleteBooking = (bookingId) => {
    setBookings(bookings.filter(booking => booking.id !== bookingId));
  };

  // Scrapbook functions
  const addMemory = (memory) => {
    setScrapbook([...scrapbook, { ...memory, id: uuidv4() }]);
  };

  const updateMemory = (memoryId, memoryData) => {
    setScrapbook(scrapbook.map(memory =>
      memory.id === memoryId ? { ...memory, ...memoryData } : memory
    ));
  };

  const deleteMemory = (memoryId) => {
    setScrapbook(scrapbook.filter(memory => memory.id !== memoryId));
  };

  const getTotalBudget = () => {
    return bookings.reduce((sum, booking) => sum + (booking.cost || 0), 0);
  };

  return (
    <HoneymoonContext.Provider value={{
      tripInfo,
      setTripInfo,
      itinerary,
      addDay,
      updateDay,
      deleteDay,
      addActivity,
      updateActivity,
      deleteActivity,
      bookings,
      addBooking,
      updateBooking,
      deleteBooking,
      scrapbook,
      addMemory,
      updateMemory,
      deleteMemory,
      getTotalBudget,
    }}>
      {children}
    </HoneymoonContext.Provider>
  );
}

export function useHoneymoon() {
  const context = useContext(HoneymoonContext);
  if (!context) {
    throw new Error('useHoneymoon must be used within a HoneymoonProvider');
  }
  return context;
}

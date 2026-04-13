-- M-flow Test Database Schema
-- Simplified version for unit testing

CREATE TABLE IF NOT EXISTS artists (
    artist_id INTEGER PRIMARY KEY,
    name VARCHAR(120) NOT NULL
);

CREATE TABLE IF NOT EXISTS albums (
    album_id INTEGER PRIMARY KEY,
    title VARCHAR(160) NOT NULL,
    artist_id INTEGER REFERENCES artists(artist_id)
);

CREATE TABLE IF NOT EXISTS tracks (
    track_id INTEGER PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    album_id INTEGER REFERENCES albums(album_id),
    genre VARCHAR(50),
    duration_ms INTEGER,
    unit_price DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS employee (
    employeeid INTEGER PRIMARY KEY,
    lastname VARCHAR(20) NOT NULL,
    firstname VARCHAR(20) NOT NULL,
    title VARCHAR(30),
    reportsto INTEGER REFERENCES employee(employeeid)
);

-- Sample data

INSERT INTO employee (employeeid, lastname, firstname, title, reportsto) VALUES
    (1, 'Adams', 'Andrew', 'General Manager', NULL),
    (2, 'Edwards', 'Nancy', 'Sales Manager', 1),
    (3, 'Peacock', 'Jane', 'Sales Support Agent', 2),
    (4, 'Park', 'Margaret', 'Sales Support Agent', 2),
    (5, 'Johnson', 'Steve', 'Sales Support Agent', 2),
    (6, 'Mitchell', 'Michael', 'IT Manager', 1),
    (7, 'King', 'Robert', 'IT Staff', 6),
    (8, 'Callahan', 'Laura', 'IT Staff', 6);

INSERT INTO artists (artist_id, name) VALUES
    (1, 'The Beatles'), (2, 'Miles Davis'), (3, 'Bach'),
    (4, 'Led Zeppelin'), (5, 'Chopin');

INSERT INTO albums (album_id, title, artist_id) VALUES
    (1, 'Abbey Road', 1), (2, 'Kind of Blue', 2),
    (3, 'Goldberg Variations', 3), (4, 'IV', 4),
    (5, 'Nocturnes', 5);

INSERT INTO tracks (track_id, name, album_id, genre, duration_ms, unit_price) VALUES
    (1, 'Come Together', 1, 'Rock', 259000, 0.99),
    (2, 'Something', 1, 'Rock', 182000, 0.99),
    (3, 'So What', 2, 'Jazz', 562000, 0.99),
    (4, 'Blue in Green', 2, 'Jazz', 338000, 0.99),
    (5, 'Aria', 3, 'Classical', 285000, 0.99),
    (6, 'Stairway to Heaven', 4, 'Rock', 482000, 0.99),
    (7, 'Nocturne Op.9 No.2', 5, 'Classical', 272000, 0.99);

(function() {
	document.write('Calling main function')
  var board, boardPositions, carFreedom, carPadding, carPositions, colours, container, detectFreedom, drag, dragEnd, dragMove, dragStart, intersection, moves, occupiedPositions, positionToX, positionToY, ref, scalar, startMoment, startXY, svg, tilePadding, timer, updateTimer;
  container = '#game';
  colours = ['#ff7fff', '#7fff7f'];
  svg = d3.select(container).append('svg').attr('width', '100%').attr('height', '100%');
  board = 6;
  scalar = 100;
  tilePadding = 16;
  carPadding = 6;
  moves = 0;
  startMoment = moment();
  updateTimer = function() {
    return d3.select('#timer').text(moment().subtract(startMoment).format('mm:ss'));
  };
  timer = setInterval(updateTimer, 1000);
  intersection = function(a1, a2) {
    return a1.filter(function(n) {
      return a2.indexOf(n) !== -1;
    });
  };

  ref = [[], 0], carFreedom = ref[0], startXY = ref[1];
  dragStart = function(d) {
    carFreedom = detectFreedom(d3.select(this));
    return startXY = d.orientation === 'horizontal' ? d.x : d.y;
  };
  dragMove = function(d) {
    var axis, car, leftBound, relativeXY, rightBound, time, xy;
    axis = d.orientation === 'horizontal' ? 'x' : 'y';
    xy = d3.event[axis];
    car = d3.select(this);
    relativeXY = xy - startXY;
    leftBound = relativeXY < 0 && Math.abs(relativeXY) < carFreedom[0] * scalar;
    rightBound = relativeXY >= 0 && relativeXY < carFreedom[1] * scalar;
    if (leftBound || rightBound) {
      return car.attr(axis, d[axis] = xy);
    } else {
      if (relativeXY < 0) {
        return car.attr(axis, d[axis] = startXY - carFreedom[0] * scalar);
      } else {
        car.attr(axis, d[axis] = startXY + carFreedom[1] * scalar);
        if (!rightBound && d.player && boardPositions(board - 1, 'vertical').indexOf(d.position + d.length - 1 + carFreedom[1]) > -1) {
          time = d3.select('#timer').text();
          clearInterval(timer);
          return d3.select('#status').text('You won! ' + time + ' and ' + (moves + 1) + ' moves');
        }
      }
    }
  };

  dragEnd = function(d) {
    var distance, newPosition, positions, xy;
    xy = d.orientation === 'horizontal' ? d.x : d.y;
    distance = Math.round((xy - startXY) / scalar);
    positions = boardPositions(d.position, d.orientation);
    newPosition = positions[positions.indexOf(d.position) + distance];
    if (Math.abs(distance) > 0) {
      d3.select('#moves').text(++moves);
    }
    return d3.select(this).attr('data-position', d.position = newPosition).transition().attr('x', d.x = positionToX(d.position)).attr('y', d.y = positionToY(d.position));
  };

  drag = d3.behavior.drag().origin(function(d) {
    return d;
  }).on('dragstart', dragStart).on('drag', dragMove).on('dragend', dragEnd);

  detectFreedom = function(car) {
    var l, lower, o, orientation, pos, positions, u, upper;
    pos = car.attr('data-position');
    orientation = car.attr('data-orientation');
    positions = boardPositions(pos, orientation);
    o = intersection(positions, occupiedPositions());
    upper = d3.min(o.filter(function(n) {
      return n > d3.max(carPositions(car));
    }));
    lower = d3.max(o.filter(function(n) {
      return n < d3.min(carPositions(car));
    }));
    if (upper) {
      u = positions.indexOf(upper) - positions.indexOf(d3.max(carPositions(car)));
    } else {
      u = board - positions.indexOf(d3.max(carPositions(car)));
    }
    l = positions.indexOf(d3.min(carPositions(car))) - positions.indexOf(lower);
    return [l - 1, u - 1];
  };

  boardPositions = function(position, orientation) {
    if (orientation === 'horizontal') {
      return d3.range(position - position % board, position - position % board + board);
    } else {
      return d3.range(position % board, Math.pow(board, 2), board);
    }
  };

  occupiedPositions = function() {
    var positions;
    positions = [];
    svg.selectAll('rect.car').each(function(d, i) {
      return positions = positions.concat(carPositions(d3.select(this)));
    });
    return positions;
  };

  carPositions = function(car) {
    var length, orientation, pos, possiblePositions, start;
    pos = parseInt(car.attr('data-position'));
    length = parseInt(car.attr('data-length'));
    orientation = car.attr('data-orientation');
    possiblePositions = boardPositions(pos, orientation);
    if (orientation === 'horizontal') {
      return possiblePositions.slice(pos % board, pos % board + length);
    } else {
      start = Math.floor(pos / board);
      return possiblePositions.slice(start, start + length);
    }
  };

  positionToX = function(position) {
    return scalar * (position % board) + carPadding;
  };

  positionToY = function(position) {
    return scalar * Math.floor(position / board) + carPadding;
  };

  d3.json('static/js/level1.json', function(error, json) {
	  document.write(json.cars)
    var car, carAttributes, cars, j, len, ref1, squareAttributes, squares;
    if (error) {
      console.warn(error);
    }
    squares = svg.append('g').attr('class', 'tiles').selectAll('rect.square').data(d3.range(Math.pow(board, 2))).enter().append('rect');
    squareAttributes = squares.attr('x', function(i) {
      return scalar * (i % board) + tilePadding;
    }).attr('y', function(i) {
      return scalar * Math.floor(i / board) + tilePadding;
    }).attr('height', scalar - tilePadding * 2).attr('width', scalar - tilePadding * 2).attr('fill', '#f4f4f7');
    ref1 = json.cars;
    for (j = 0, len = ref1.length; j < len; j++) {
      car = ref1[j];
      car.x = positionToX(car.position);
      car.y = positionToY(car.position);
    }
    cars = svg.append('g').attr('class', 'cars').selectAll('rect.car').data(json.cars).enter().append('rect');
    return carAttributes = cars.attr('class', 'car').attr('data-position', function(d) {
      return d.position;
    }).attr('data-length', function(d) {
      return d.length;
    }).attr('data-orientation', function(d) {
      return d.orientation;
    }).attr('x', function(d) {
      return d.x;
    }).attr('y', function(d) {
      return d.y;
    }).attr('height', function(d) {
      return (d.orientation === 'vertical' ? scalar * d.length : scalar) - carPadding * 2;
    }).attr('width', function(d) {
      return (d.orientation === 'horizontal' ? scalar * d.length : scalar) - carPadding * 2;
    }).attr('fill', function(d) {
      if (d.player) {
        return colours[1];
      } else {
        return colours[0];
      }
    }).call(drag);
  });
}).call(this);

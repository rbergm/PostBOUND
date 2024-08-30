SELECT COUNT(*)
FROM votes AS v, posts AS p, users AS u
WHERE v.PostId = p.Id
  AND v.UserId = u.Id
  AND v.CreationDate <= CAST('2014-09-12 00:00:00' AS timestamp)
  AND p.Score >= -1
  AND p.CreationDate >= CAST('2010-10-21 13:21:24' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-09 15:12:22' AS timestamp)
  AND u.UpVotes >= 0
  AND u.CreationDate >= CAST('2010-07-27 17:15:57' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-03 12:47:42' AS timestamp);

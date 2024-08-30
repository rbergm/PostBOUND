SELECT COUNT(*)
FROM postHistory AS ph, posts AS p, users AS u
WHERE p.OwnerUserId = u.Id
  AND ph.UserId = u.Id
  AND ph.CreationDate >= CAST('2011-05-20 18:43:03' AS timestamp)
  AND p.FavoriteCount <= 5
  AND u.Views >= 0
  AND u.UpVotes >= 0
  AND u.CreationDate >= CAST('2010-11-27 21:46:49' AS timestamp)
  AND u.CreationDate <= CAST('2014-08-18 13:00:22' AS timestamp);
